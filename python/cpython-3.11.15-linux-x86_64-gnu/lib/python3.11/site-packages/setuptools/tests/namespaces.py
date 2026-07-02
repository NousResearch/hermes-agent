import ast
import json
import textwrap
from pathlib import Path


def iter_namespace_pkgs(namespace):
    parts = namespace.split(".")
    for i in range(len(parts)):
        yield ".".join(parts[: i + 1])


def build_namespace_package(tmpdir, name, version="1.0", impl="pkg_resources"):
    src_dir = tmpdir / name
    src_dir.mkdir()
    setup_py = src_dir / 'setup.py'
    namespace, _, rest = name.rpartition('.')
    namespaces = list(iter_namespace_pkgs(namespace))
    setup_args = {
        "name": name,
        "version": version,
        "packages": namespaces,
    }

    if impl == "pkg_resources":
        tmpl = '__import__("pkg_resources").declare_namespace(__name__)'
        setup_args["namespace_packages"] = namespaces
    elif impl == "pkgutil":
        tmpl = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    else:
        raise ValueError(f"Cannot recognise {impl=} when creating namespaces")

    args = json.dumps(setup_args, indent=4)
    assert ast.literal_eval(args)  # ensure it is valid Python

    script = textwrap.dedent(
        """\
        import setuptools
        args = {args}
        setuptools.setup(**args)
        """
    ).format(args=args)
    setup_py.write_text(script, encoding='utf-8')

    ns_pkg_dir = Path(src_dir, namespace.replace(".", "/"))
    ns_pkg_dir.mkdir(parents=True)

    for ns in namespaces:
        pkg_init = src_dir / ns.replace(".", "/") / '__init__.py'
        pkg_init.write_text(tmpl, encoding='utf-8')

    pkg_mod = ns_pkg_dir / (rest + '.py')
    some_functionality = 'name = {rest!r}'.format(**locals())
    pkg_mod.write_text(some_functionality, encoding='utf-8')
    return src_dir


def build_pep420_namespace_package(tmpdir, name):
    src_dir = tmpdir / name
    src_dir.mkdir()
    pyproject = src_dir / "pyproject.toml"
    namespace, _, rest = name.rpartition(".")
    script = f"""\
        [build-system]
        requires = ["setuptools"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{name}"
        version = "3.14159"
        """
    pyproject.write_text(textwrap.dedent(script), encoding='utf-8')
    ns_pkg_dir = Path(src_dir, namespace.replace(".", "/"))
    ns_pkg_dir.mkdir(parents=True)
    pkg_mod = ns_pkg_dir / (rest + ".py")
    some_functionality = f"name = {rest!r}"
    pkg_mod.write_text(some_functionality, encoding='utf-8')
    return src_dir


def make_site_dir(target):
    """
    Add a sitecustomize.py module in target to cause
    target to be added to site dirs such that .pth files
    are processed there.
    """
    sc = target / 'sitecustomize.py'
    target_str = str(target)
    tmpl = '__import__("site").addsitedir({target_str!r})'
    sc.write_text(tmpl.format(**locals()), encoding='utf-8')
