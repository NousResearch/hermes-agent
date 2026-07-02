import email.message
import email.policy
import re
import textwrap

from ._text import FoldedCase


class RawPolicy(email.policy.EmailPolicy):
    def fold(self, name, value):
        folded = self.linesep.join(
            textwrap
            .indent(value, prefix=' ' * 8, predicate=lambda line: True)
            .lstrip()
            .splitlines()
        )
        return f'{name}: {folded}{self.linesep}'


class Message(email.message.Message):
    r"""
    Specialized Message subclass to handle metadata naturally.

    Reads values that may have newlines in them and converts the
    payload to the Description.

    >>> msg_text = textwrap.dedent('''
    ...     Name: Foo
    ...     Version: 3.0
    ...     License: blah
    ...             de-blah
    ...     <BLANKLINE>
    ...     First line of description.
    ...     Second line of description.
    ...     <BLANKLINE>
    ...     Fourth line!
    ...     ''').lstrip().replace('<BLANKLINE>', '')
    >>> msg = Message(email.message_from_string(msg_text))
    >>> msg['Description']
    'First line of description.\nSecond line of description.\n\nFourth line!\n'

    Message should render even if values contain newlines.

    >>> print(msg)
    Name: Foo
    Version: 3.0
    License: blah
            de-blah
    Description: First line of description.
            Second line of description.
    <BLANKLINE>
            Fourth line!
    <BLANKLINE>
    <BLANKLINE>
    """

    multiple_use_keys = set(
        map(
            FoldedCase,
            [
                'Classifier',
                'Obsoletes-Dist',
                'Platform',
                'Project-URL',
                'Provides-Dist',
                'Provides-Extra',
                'Requires-Dist',
                'Requires-External',
                'Supported-Platform',
                'Dynamic',
            ],
        )
    )
    """
    Keys that may be indicated multiple times per PEP 566.
    """

    def __new__(cls, orig: email.message.Message):
        res = super().__new__(cls)
        vars(res).update(vars(orig))
        return res

    def __init__(self, *args, **kwargs):
        self._headers = self._repair_headers()

    # suppress spurious error from mypy
    def __iter__(self):
        return super().__iter__()

    def __getitem__(self, item):
        """
        Override parent behavior to typical dict behavior.

        ``email.message.Message`` will emit None values for missing
        keys. Typical mappings, including this ``Message``, will raise
        a key error for missing keys.

        Ref python/importlib_metadata#371.
        """
        res = super().__getitem__(item)
        if res is None:
            raise KeyError(item)
        return res

    def _repair_headers(self):
        def redent(value):
            "Correct for RFC822 indentation"
            indent = ' ' * 8
            if not value or '\n' + indent not in value:
                return value
            return textwrap.dedent(indent + value)

        headers = [(key, redent(value)) for key, value in vars(self)['_headers']]
        if self._payload:
            headers.append(('Description', self.get_payload()))
            self.set_payload('')
        return headers

    def as_string(self):
        return super().as_string(policy=RawPolicy())

    @property
    def json(self):
        """
        Convert PackageMetadata to a JSON-compatible format
        per PEP 0566.
        """

        def transform(key):
            value = self.get_all(key) if key in self.multiple_use_keys else self[key]
            if key == 'Keywords':
                value = re.split(r'\s+', value)
            tk = key.lower().replace('-', '_')
            return tk, value

        return dict(map(transform, map(FoldedCase, self)))
