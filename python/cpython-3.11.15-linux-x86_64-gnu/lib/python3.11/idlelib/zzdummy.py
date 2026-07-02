"""Example extension, also used for testing.

See extend.txt for more details on creating an extension.
See config-extension.def for configuring an extension.
"""

from idlelib.config import idleConf
from functools import wraps


def format_selection(format_line):
    "Apply a formatting function to all of the selected lines."

    @wraps(format_line)
    def apply(self, event=None):
        head, tail, chars, lines = self.formatter.get_region()
        for pos in range(len(lines) - 1):
            line = lines[pos]
            lines[pos] = format_line(self, line)
        self.formatter.set_region(head, tail, chars, lines)
        return 'break'

    return apply


class ZzDummy:
    """Prepend or remove initial text from selected lines."""

    # Extend the format menu.
    menudefs = [
        ('format', [
            ('Z in', '<<z-in>>'),
            ('Z out', '<<z-out>>'),
        ] )
    ]

    def __init__(self, editwin):
        "Initialize the settings for this extension."
        self.editwin = editwin
        self.text = editwin.text
        self.formatter = editwin.fregion

    @classmethod
    def reload(cls):
        "Load class variables from config."
        cls.ztext = idleConf.GetOption('extensions', 'ZzDummy', 'z-text')

    @format_selection
    def z_in_event(self, line):
        """Insert text at the beginning of each selected line.

        This is bound to the <<z-in>> virtual event when the extensions
        are loaded.
        """
        return f'{self.ztext}{line}'

    @format_selection
    def z_out_event(self, line):
        """Remove specific text from the beginning of each selected line.

        This is bound to the <<z-out>> virtual event when the extensions
        are loaded.
        """
        zlength = 0 if not line.startswith(self.ztext) else len(self.ztext)
        return line[zlength:]


ZzDummy.reload()


if __name__ == "__main__":
    import unittest
    unittest.main('idlelib.idle_test.test_zzdummy', verbosity=2, exit=False)
