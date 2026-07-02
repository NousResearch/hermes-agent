"""
Load setuptools configuration from ``setup.cfg`` files.

**API will be made private in the future**

To read project metadata, consider using
``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).
For simple scenarios, you can also try parsing the file directly
with the help of ``configparser``.
"""

from __future__ import annotations

import contextlib
import functools
import os
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, TypeVar, cast

from packaging.markers import default_environment as marker_env
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from .. import _static
from .._path import StrPath
from ..errors import FileError, OptionError
from ..warnings import SetuptoolsDeprecationWarning
from . import expand

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from setuptools.dist import Distribution

    from distutils.dist import DistributionMetadata

SingleCommandOptions: TypeAlias = dict[str, tuple[str, Any]]
"""Dict that associate the name of the options of a particular command to a
tuple. The first element of the tuple indicates the origin of the option value
(e.g. the name of the configuration file where it was read from),
while the second element of the tuple is the option value itself
"""
AllCommandOptions: TypeAlias = dict[str, SingleCommandOptions]
"""cmd name => its options"""
Target = TypeVar("Target", "Distribution", "DistributionMetadata")


def read_configuration(
    filepath: StrPath, find_others: bool = False, ignore_option_errors: bool = False
) -> dict:
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file
        to get options from.

    :param bool find_others: Whether to search for other configuration files
        which could be on in various places.

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :rtype: dict
    """
    from setuptools.dist import Distribution

    dist = Distribution()
    filenames = dist.find_config_files() if find_others else []
    handlers = _apply(dist, filepath, filenames, ignore_option_errors)
    return configuration_to_dict(handlers)


def apply_configuration(dist: Distribution, filepath: StrPath) -> Distribution:
    """Apply the configuration from a ``setup.cfg`` file into an existing
    distribution object.
    """
    _apply(dist, filepath)
    dist._finalize_requires()
    return dist


def _apply(
    dist: Distribution,
    filepath: StrPath,
    other_files: Iterable[StrPath] = (),
    ignore_option_errors: bool = False,
) -> tuple[ConfigMetadataHandler, ConfigOptionsHandler]:
    """Read configuration from ``filepath`` and applies to the ``dist`` object."""
    from setuptools.dist import _Distribution

    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise FileError(f'Configuration file {filepath} does not exist.')

    current_directory = os.getcwd()
    os.chdir(os.path.dirname(filepath))
    filenames = [*other_files, filepath]

    try:
        # TODO: Temporary cast until mypy 1.12 is released with upstream fixes from typeshed
        _Distribution.parse_config_files(dist, filenames=cast(list[str], filenames))
        handlers = parse_configuration(
            dist, dist.command_options, ignore_option_errors=ignore_option_errors
        )
        dist._finalize_license_files()
    finally:
        os.chdir(current_directory)

    return handlers


def _get_option(target_obj: Distribution | DistributionMetadata, key: str):
    """
    Given a target object and option key, get that option from
    the target object, either through a get_{key} method or
    from an attribute directly.
    """
    getter_name = f'get_{key}'
    by_attribute = functools.partial(getattr, target_obj, key)
    getter = getattr(target_obj, getter_name, by_attribute)
    return getter()


def configuration_to_dict(
    handlers: Iterable[
        ConfigHandler[Distribution] | ConfigHandler[DistributionMetadata]
    ],
) -> dict:
    """Returns configuration data gathered by given handlers as a dict.

    :param Iterable[ConfigHandler] handlers: Handlers list,
        usually from parse_configuration()

    :rtype: dict
    """
    config_dict: dict = defaultdict(dict)

    for handler in handlers:
        for option in handler.set_options:
            value = _get_option(handler.target_obj, option)
            config_dict[handler.section_prefix][option] = value

    return config_dict


def parse_configuration(
    distribution: Distribution,
    command_options: AllCommandOptions,
    ignore_option_errors: bool = False,
) -> tuple[ConfigMetadataHandler, ConfigOptionsHandler]:
    """Performs additional parsing of configuration options
    for a distribution.

    Returns a list of used option handlers.

    :param Distribution distribution:
    :param dict command_options:
    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.
    :rtype: list
    """
    with expand.EnsurePackagesDiscovered(distribution) as ensure_discovered:
        options = ConfigOptionsHandler(
            distribution,
            command_options,
            ignore_option_errors,
            ensure_discovered,
        )

        options.parse()
        if not distribution.package_dir:
            distribution.package_dir = options.package_dir  # Filled by `find_packages`

        meta = ConfigMetadataHandler(
            distribution.metadata,
            command_options,
            ignore_option_errors,
            ensure_discovered,
            distribution.package_dir,
            distribution.src_root,
        )
        meta.parse()
        distribution._referenced_files.update(
            options._referenced_files, meta._referenced_files
        )

    return meta, options


def _warn_accidental_env_marker_misconfig(label: str, orig_value: str, parsed: list):
    """Because users sometimes misinterpret this configuration:

    [options.extras_require]
    foo = bar;python_version<"4"

    It looks like one requirement with an environment marker
    but because there is no newline, it's parsed as two requirements
    with a semicolon as separator.

    Therefore, if:
        * input string does not contain a newline AND
        * parsed result contains two requirements AND
        * parsing of the two parts from the result ("<first>;<second>")
        leads in a valid Requirement with a valid marker
    a UserWarning is shown to inform the user about the possible problem.
    """
    if "\n" in orig_value or len(parsed) != 2:
        return

    markers = marker_env().keys()

    try:
        req = Requirement(parsed[1])
        if req.name in markers:
            _AmbiguousMarker.emit(field=label, req=parsed[1])
    except InvalidRequirement as ex:
        if any(parsed[1].startswith(marker) for marker in markers):
            msg = _AmbiguousMarker.message(field=label, req=parsed[1])
            raise InvalidRequirement(msg) from ex


class ConfigHandler(Generic[Target]):
    """Handles metadata supplied in configuration files."""

    section_prefix: str
    """Prefix for config sections handled by this handler.
    Must be provided by class heirs.

    """

    aliases: ClassVar[dict[str, str]] = {}
    """Options aliases.
    For compatibility with various packages. E.g.: d2to1 and pbr.
    Note: `-` in keys is replaced with `_` by config parser.

    """

    def __init__(
        self,
        target_obj: Target,
        options: AllCommandOptions,
        ignore_option_errors,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ) -> None:
        self.ignore_option_errors = ignore_option_errors
        self.target_obj: Target = target_obj
        self.sections = dict(self._section_options(options))
        self.set_options: list[str] = []
        self.ensure_discovered = ensure_discovered
        self._referenced_files = set[str]()
        """After parsing configurations, this property will enumerate
        all files referenced by the "file:" directive. Private API for setuptools only.
        """

    @classmethod
    def _section_options(
        cls, options: AllCommandOptions
    ) -> Iterator[tuple[str, SingleCommandOptions]]:
        for full_name, value in options.items():
            pre, _sep, name = full_name.partition(cls.section_prefix)
            if pre:
                continue
            yield name.lstrip('.'), value

    @property
    @abstractmethod
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        raise NotImplementedError(
            f'{self.__class__.__name__} must provide .parsers property'
        )

    def __setitem__(self, option_name, value) -> None:
        target_obj = self.target_obj

        # Translate alias into real name.
        option_name = self.aliases.get(option_name, option_name)

        try:
            current_value = getattr(target_obj, option_name)
        except AttributeError as e:
            raise KeyError(option_name) from e

        if current_value:
            # Already inhabited. Skipping.
            return

        try:
            parsed = self.parsers.get(option_name, lambda x: x)(value)
        except (Exception,) * self.ignore_option_errors:
            return

        simple_setter = functools.partial(target_obj.__setattr__, option_name)
        setter = getattr(target_obj, f"set_{option_name}", simple_setter)
        setter(parsed)

        self.set_options.append(option_name)

    @classmethod
    def _parse_list(cls, value, separator=','):
        """Represents value as a list.

        Value is split either by separator (defaults to comma) or by lines.

        :param value:
        :param separator: List items separator character.
        :rtype: list
        """
        if isinstance(value, list):  # _get_parser_compound case
            return value

        if '\n' in value:
            value = value.splitlines()
        else:
            value = value.split(separator)

        return [chunk.strip() for chunk in value if chunk.strip()]

    @classmethod
    def _parse_dict(cls, value):
        """Represents value as a dict.

        :param value:
        :rtype: dict
        """
        separator = '='
        result = {}
        for line in cls._parse_list(value):
            key, sep, val = line.partition(separator)
            if sep != separator:
                raise OptionError(f"Unable to parse option value to dict: {value}")
            result[key.strip()] = val.strip()

        return result

    @classmethod
    def _parse_bool(cls, value):
        """Represents value as boolean.

        :param value:
        :rtype: bool
        """
        value = value.lower()
        return value in ('1', 'true', 'yes')

    @classmethod
    def _exclude_files_parser(cls, key):
        """Returns a parser function to make sure field inputs
        are not files.

        Parses a value after getting the key so error messages are
        more informative.

        :param key:
        :rtype: callable
        """

        def parser(value):
            exclude_directive = 'file:'
            if value.startswith(exclude_directive):
                raise ValueError(
                    f'Only strings are accepted for the {key} field, '
                    'files are not accepted'
                )
            return _static.Str(value)

        return parser

    def _parse_file(self, value, root_dir: StrPath | None):
        """Represents value as a string, allowing including text
        from nearest files using `file:` directive.

        Directive is sandboxed and won't reach anything outside
        directory with setup.py.

        Examples:
            file: README.rst, CHANGELOG.md, src/file.txt

        :param str value:
        :rtype: str
        """
        include_directive = 'file:'

        if not isinstance(value, str):
            return value

        if not value.startswith(include_directive):
            return _static.Str(value)

        spec = value[len(include_directive) :]
        filepaths = [path.strip() for path in spec.split(',')]
        self._referenced_files.update(filepaths)
        # XXX: Is marking as static contents coming from files too optimistic?
        return _static.Str(expand.read_files(filepaths, root_dir))

    def _parse_attr(self, value, package_dir, root_dir: StrPath):
        """Represents value as a module attribute.

        Examples:
            attr: package.attr
            attr: package.module.attr

        :param str value:
        :rtype: str
        """
        attr_directive = 'attr:'
        if not value.startswith(attr_directive):
            return _static.Str(value)

        attr_desc = value.replace(attr_directive, '')

        # Make sure package_dir is populated correctly, so `attr:` directives can work
        package_dir.update(self.ensure_discovered.package_dir)
        return expand.read_attr(attr_desc, package_dir, root_dir)

    @classmethod
    def _get_parser_compound(cls, *parse_methods):
        """Returns parser function to represents value as a list.

        Parses a value applying given methods one after another.

        :param parse_methods:
        :rtype: callable
        """

        def parse(value):
            parsed = value

            for method in parse_methods:
                parsed = method(parsed)

            return parsed

        return parse

    @classmethod
    def _parse_section_to_dict_with_key(cls, section_options, values_parser):
        """Parses section options into a dictionary.

        Applies a given parser to each option in a section.

        :param dict section_options:
        :param callable values_parser: function with 2 args corresponding to key, value
        :rtype: dict
        """
        value = {}
        for key, (_, val) in section_options.items():
            value[key] = values_parser(key, val)
        return value

    @classmethod
    def _parse_section_to_dict(cls, section_options, values_parser=None):
        """Parses section options into a dictionary.

        Optionally applies a given parser to each value.

        :param dict section_options:
        :param callable values_parser: function with 1 arg corresponding to option value
        :rtype: dict
        """
        parser = (lambda _, v: values_parser(v)) if values_parser else (lambda _, v: v)
        return cls._parse_section_to_dict_with_key(section_options, parser)

    def parse_section(self, section_options) -> None:
        """Parses configuration file section.

        :param dict section_options:
        """
        for name, (_, value) in section_options.items():
            with contextlib.suppress(KeyError):
                # Keep silent for a new option may appear anytime.
                self[name] = value

    def parse(self) -> None:
        """Parses configuration file items from one
        or more related sections.

        """
        for section_name, section_options in self.sections.items():
            method_postfix = ''
            if section_name:  # [section.option] variant
                method_postfix = f"_{section_name}"

            section_parser_method: Callable | None = getattr(
                self,
                # Dots in section names are translated into dunderscores.
                f'parse_section{method_postfix}'.replace('.', '__'),
                None,
            )

            if section_parser_method is None:
                raise OptionError(
                    "Unsupported distribution option section: "
                    f"[{self.section_prefix}.{section_name}]"
                )

            section_parser_method(section_options)

    def _deprecated_config_handler(self, func, msg, **kw):
        """this function will wrap around parameters that are deprecated

        :param msg: deprecation message
        :param func: function to be wrapped around
        """

        @wraps(func)
        def config_handler(*args, **kwargs):
            kw.setdefault("stacklevel", 2)
            _DeprecatedConfig.emit("Deprecated config in `setup.cfg`", msg, **kw)
            return func(*args, **kwargs)

        return config_handler


class ConfigMetadataHandler(ConfigHandler["DistributionMetadata"]):
    section_prefix = 'metadata'

    aliases = {
        'home_page': 'url',
        'summary': 'description',
        'classifier': 'classifiers',
        'platform': 'platforms',
    }

    strict_mode = False
    """We need to keep it loose, to be partially compatible with
    `pbr` and `d2to1` packages which also uses `metadata` section.

    """

    def __init__(
        self,
        target_obj: DistributionMetadata,
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
        package_dir: dict | None = None,
        root_dir: StrPath | None = os.curdir,
    ) -> None:
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.package_dir = package_dir
        self.root_dir = root_dir

    @property
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        parse_list_static = self._get_parser_compound(self._parse_list, _static.List)
        parse_dict_static = self._get_parser_compound(self._parse_dict, _static.Dict)
        parse_file = partial(self._parse_file, root_dir=self.root_dir)
        exclude_files_parser = self._exclude_files_parser

        return {
            'author': _static.Str,
            'author_email': _static.Str,
            'maintainer': _static.Str,
            'maintainer_email': _static.Str,
            'platforms': parse_list_static,
            'keywords': parse_list_static,
            'provides': parse_list_static,
            'obsoletes': parse_list_static,
            'classifiers': self._get_parser_compound(parse_file, parse_list_static),
            'license': exclude_files_parser('license'),
            'license_files': parse_list_static,
            'description': parse_file,
            'long_description': parse_file,
            'long_description_content_type': _static.Str,
            'version': self._parse_version,  # Cannot be marked as dynamic
            'url': _static.Str,
            'project_urls': parse_dict_static,
        }

    def _parse_version(self, value):
        """Parses `version` option value.

        :param value:
        :rtype: str

        """
        version = self._parse_file(value, self.root_dir)

        if version != value:
            version = version.strip()
            # Be strict about versions loaded from file because it's easy to
            # accidentally include newlines and other unintended content
            try:
                Version(version)
            except InvalidVersion as e:
                raise OptionError(
                    f'Version loaded from {value} does not '
                    f'comply with PEP 440: {version}'
                ) from e

            return version

        return expand.version(self._parse_attr(value, self.package_dir, self.root_dir))


class ConfigOptionsHandler(ConfigHandler["Distribution"]):
    section_prefix = 'options'

    def __init__(
        self,
        target_obj: Distribution,
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ) -> None:
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.root_dir = target_obj.src_root
        self.package_dir: dict[str, str] = {}  # To be filled by `find_packages`

    @classmethod
    def _parse_list_semicolon(cls, value):
        return cls._parse_list(value, separator=';')

    def _parse_file_in_root(self, value):
        return self._parse_file(value, root_dir=self.root_dir)

    def _parse_requirements_list(self, label: str, value: str):
        # Parse a requirements list, either by reading in a `file:`, or a list.
        parsed = self._parse_list_semicolon(self._parse_file_in_root(value))
        _warn_accidental_env_marker_misconfig(label, value, parsed)
        # Filter it to only include lines that are not comments. `parse_list`
        # will have stripped each line and filtered out empties.
        return _static.List(line for line in parsed if not line.startswith("#"))
        # ^-- Use `_static.List` to mark a non-`Dynamic` Core Metadata

    @property
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_bool = self._parse_bool
        parse_cmdclass = self._parse_cmdclass

        return {
            'zip_safe': parse_bool,
            'include_package_data': parse_bool,
            'package_dir': self._parse_dict,
            'scripts': parse_list,
            'eager_resources': parse_list,
            'dependency_links': parse_list,
            'namespace_packages': self._deprecated_config_handler(
                parse_list,
                "The namespace_packages parameter is deprecated, "
                "consider using implicit namespaces instead (PEP 420).",
                # TODO: define due date, see setuptools.dist:check_nsp.
            ),
            'install_requires': partial(  # Core Metadata
                self._parse_requirements_list, "install_requires"
            ),
            'setup_requires': self._parse_list_semicolon,
            'packages': self._parse_packages,
            'entry_points': self._parse_file_in_root,
            'py_modules': parse_list,
            'python_requires': _static.SpecifierSet,  # Core Metadata
            'cmdclass': parse_cmdclass,
        }

    def _parse_cmdclass(self, value):
        package_dir = self.ensure_discovered.package_dir
        return expand.cmdclass(self._parse_dict(value), package_dir, self.root_dir)

    def _parse_packages(self, value):
        """Parses `packages` option value.

        :param value:
        :rtype: list
        """
        find_directives = ['find:', 'find_namespace:']
        trimmed_value = value.strip()

        if trimmed_value not in find_directives:
            return self._parse_list(value)

        # Read function arguments from a dedicated section.
        find_kwargs = self.parse_section_packages__find(
            self.sections.get('packages.find', {})
        )

        find_kwargs.update(
            namespaces=(trimmed_value == find_directives[1]),
            root_dir=self.root_dir,
            fill_package_dir=self.package_dir,
        )

        return expand.find_packages(**find_kwargs)

    def parse_section_packages__find(self, section_options):
        """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
        section_data = self._parse_section_to_dict(section_options, self._parse_list)

        valid_keys = ['where', 'include', 'exclude']
        find_kwargs = {k: v for k, v in section_data.items() if k in valid_keys and v}

        where = find_kwargs.get('where')
        if where is not None:
            find_kwargs['where'] = where[0]  # cast list to single val

        return find_kwargs

    def parse_section_entry_points(self, section_options) -> None:
        """Parses `entry_points` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['entry_points'] = parsed

    def _parse_package_data(self, section_options):
        package_data = self._parse_section_to_dict(section_options, self._parse_list)
        return expand.canonic_package_data(package_data)

    def parse_section_package_data(self, section_options) -> None:
        """Parses `package_data` configuration file section.

        :param dict section_options:
        """
        self['package_data'] = self._parse_package_data(section_options)

    def parse_section_exclude_package_data(self, section_options) -> None:
        """Parses `exclude_package_data` configuration file section.

        :param dict section_options:
        """
        self['exclude_package_data'] = self._parse_package_data(section_options)

    def parse_section_extras_require(self, section_options) -> None:  # Core Metadata
        """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict_with_key(
            section_options,
            lambda k, v: self._parse_requirements_list(f"extras_require[{k}]", v),
        )

        self['extras_require'] = _static.Dict(parsed)
        # ^-- Use `_static.Dict` to mark a non-`Dynamic` Core Metadata

    def parse_section_data_files(self, section_options) -> None:
        """Parses `data_files` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['data_files'] = expand.canonic_data_files(parsed, self.root_dir)


class _AmbiguousMarker(SetuptoolsDeprecationWarning):
    _SUMMARY = "Ambiguous requirement marker."
    _DETAILS = """
    One of the parsed requirements in `{field}` looks like a valid environment marker:

        {req!r}

    Please make sure that the configuration file is correct.
    You can use dangling lines to avoid this problem.
    """
    _SEE_DOCS = "userguide/declarative_config.html#opt-2"
    # TODO: should we include due_date here? Initially introduced in 6 Aug 2022.
    # Does this make sense with latest version of packaging?

    @classmethod
    def message(cls, **kw):
        docs = f"https://setuptools.pypa.io/en/latest/{cls._SEE_DOCS}"
        return cls._format(cls._SUMMARY, cls._DETAILS, see_url=docs, format_args=kw)


class _DeprecatedConfig(SetuptoolsDeprecationWarning):
    _SEE_DOCS = "userguide/declarative_config.html"
