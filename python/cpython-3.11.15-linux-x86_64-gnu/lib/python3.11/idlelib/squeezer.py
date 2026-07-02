"""An IDLE extension to avoid having very long texts printed in the shell.

A common problem in IDLE's interactive shell is printing of large amounts of
text into the shell. This makes looking at the previous history difficult.
Worse, this can cause IDLE to become very slow, even to the point of being
completely unusable.

This extension will automatically replace long texts with a small button.
Double-clicking this button will remove it and insert the original text instead.
Middle-clicking will copy the text to the clipboard. Right-clicking will open
the text in a separate viewing window.

Additionally, any output can be manually "squeezed" by the user. This includes
output written to the standard error stream ("stderr"), such as exception
messages and their tracebacks.
"""
import re

import tkinter as tk
from tkinter import messagebox

from idlelib.config import idleConf
from idlelib.textview import view_text
from idlelib.tooltip import Hovertip
from idlelib import macosx


def count_lines_with_wrapping(s, linewidth=80):
    """Count the number of lines in a given string.

    Lines are counted as if the string was wrapped so that lines are never over
    linewidth characters long.

    Tabs are considered tabwidth characters long.
    """
    tabwidth = 8  # Currently always true in Shell.
    pos = 0
    linecount = 1
    current_column = 0

    for m in re.finditer(r"[\t\n]", s):
        # Process the normal chars up to tab or newline.
        numchars = m.start() - pos
        pos += numchars
        current_column += numchars

        # Deal with tab or newline.
        if s[pos] == '\n':
            # Avoid the `current_column == 0` edge-case, and while we're
            # at it, don't bother adding 0.
            if current_column > linewidth:
                # If the current column was exactly linewidth, divmod
                # would give (1,0), even though a new line hadn't yet
                # been started. The same is true if length is any exact
                # multiple of linewidth. Therefore, subtract 1 before
                # dividing a non-empty line.
                linecount += (current_column - 1) // linewidth
            linecount += 1
            current_column = 0
        else:
            assert s[pos] == '\t'
            current_column += tabwidth - (current_column % tabwidth)

            # If a tab passes the end of the line, consider the entire
            # tab as being on the next line.
            if current_column > linewidth:
                linecount += 1
                current_column = tabwidth

        pos += 1 # After the tab or newline.

    # Process remaining chars (no more tabs or newlines).
    current_column += len(s) - pos
    # Avoid divmod(-1, linewidth).
    if current_column > 0:
        linecount += (current_column - 1) // linewidth
    else:
        # Text ended with newline; don't count an extra line after it.
        linecount -= 1

    return linecount


class ExpandingButton(tk.Button):
    """Class for the "squeezed" text buttons used by Squeezer

    These buttons are displayed inside a Tk Text widget in place of text. A
    user can then use the button to replace it with the original text, copy
    the original text to the clipboard or view the original text in a separate
    window.

    Each button is tied to a Squeezer instance, and it knows to update the
    Squeezer instance when it is expanded (and therefore removed).
    """
    def __init__(self, s, tags, numoflines, squeezer):
        self.s = s
        self.tags = tags
        self.numoflines = numoflines
        self.squeezer = squeezer
        self.editwin = editwin = squeezer.editwin
        self.text = text = editwin.text
        # The base Text widget is needed to change text before iomark.
        self.base_text = editwin.per.bottom

        line_plurality = "lines" if numoflines != 1 else "line"
        button_text = f"Squeezed text ({numoflines} {line_plurality})."
        tk.Button.__init__(self, text, text=button_text,
                           background="#FFFFC0", activebackground="#FFFFE0")

        button_tooltip_text = (
            "Double-click to expand, right-click for more options."
        )
        Hovertip(self, button_tooltip_text, hover_delay=80)

        self.bind("<Double-Button-1>", self.expand)
        if macosx.isAquaTk():
            # AquaTk defines <2> as the right button, not <3>.
            self.bind("<Button-2>", self.context_menu_event)
        else:
            self.bind("<Button-3>", self.context_menu_event)
        self.selection_handle(  # X windows only.
            lambda offset, length: s[int(offset):int(offset) + int(length)])

        self.is_dangerous = None
        self.after_idle(self.set_is_dangerous)

    def set_is_dangerous(self):
        dangerous_line_len = 50 * self.text.winfo_width()
        self.is_dangerous = (
            self.numoflines > 1000 or
            len(self.s) > 50000 or
            any(
                len(line_match.group(0)) >= dangerous_line_len
                for line_match in re.finditer(r'[^\n]+', self.s)
            )
        )

    def expand(self, event=None):
        """expand event handler

        This inserts the original text in place of the button in the Text
        widget, removes the button and updates the Squeezer instance.

        If the original text is dangerously long, i.e. expanding it could
        cause a performance degradation, ask the user for confirmation.
        """
        if self.is_dangerous is None:
            self.set_is_dangerous()
        if self.is_dangerous:
            confirm = messagebox.askokcancel(
                title="Expand huge output?",
                message="\n\n".join([
                    "The squeezed output is very long: %d lines, %d chars.",
                    "Expanding it could make IDLE slow or unresponsive.",
                    "It is recommended to view or copy the output instead.",
                    "Really expand?"
                ]) % (self.numoflines, len(self.s)),
                default=messagebox.CANCEL,
                parent=self.text)
            if not confirm:
                return "break"

        index = self.text.index(self)
        self.base_text.insert(index, self.s, self.tags)
        self.base_text.delete(self)
        self.editwin.on_squeezed_expand(index, self.s, self.tags)
        self.squeezer.expandingbuttons.remove(self)

    def copy(self, event=None):
        """copy event handler

        Copy the original text to the clipboard.
        """
        self.clipboard_clear()
        self.clipboard_append(self.s)

    def view(self, event=None):
        """view event handler

        View the original text in a separate text viewer window.
        """
        view_text(self.text, "Squeezed Output Viewer", self.s,
                  modal=False, wrap='none')

    rmenu_specs = (
        # Item structure: (label, method_name).
        ('copy', 'copy'),
        ('view', 'view'),
    )

    def context_menu_event(self, event):
        self.text.mark_set("insert", "@%d,%d" % (event.x, event.y))
        rmenu = tk.Menu(self.text, tearoff=0)
        for label, method_name in self.rmenu_specs:
            rmenu.add_command(label=label, command=getattr(self, method_name))
        rmenu.tk_popup(event.x_root, event.y_root)
        return "break"


class Squeezer:
    """Replace long outputs in the shell with a simple button.

    This avoids IDLE's shell slowing down considerably, and even becoming
    completely unresponsive, when very long outputs are written.
    """
    @classmethod
    def reload(cls):
        """Load class variables from config."""
        cls.auto_squeeze_min_lines = idleConf.GetOption(
            "main", "PyShell", "auto-squeeze-min-lines",
            type="int", default=50,
        )

    def __init__(self, editwin):
        """Initialize settings for Squeezer.

        editwin is the shell's Editor window.
        self.text is the editor window text widget.
        self.base_test is the actual editor window Tk text widget, rather than
            EditorWindow's wrapper.
        self.expandingbuttons is the list of all buttons representing
            "squeezed" output.
        """
        self.editwin = editwin
        self.text = text = editwin.text

        # Get the base Text widget of the PyShell object, used to change
        # text before the iomark. PyShell deliberately disables changing
        # text before the iomark via its 'text' attribute, which is
        # actually a wrapper for the actual Text widget. Squeezer,
        # however, needs to make such changes.
        self.base_text = editwin.per.bottom

        # Twice the text widget's border width and internal padding;
        # pre-calculated here for the get_line_width() method.
        self.window_width_delta = 2 * (
            int(text.cget('border')) +
            int(text.cget('padx'))
        )

        self.expandingbuttons = []

        # Replace the PyShell instance's write method with a wrapper,
        # which inserts an ExpandingButton instead of a long text.
        def mywrite(s, tags=(), write=editwin.write):
            # Only auto-squeeze text which has just the "stdout" tag.
            if tags != "stdout":
                return write(s, tags)

            # Only auto-squeeze text with at least the minimum
            # configured number of lines.
            auto_squeeze_min_lines = self.auto_squeeze_min_lines
            # First, a very quick check to skip very short texts.
            if len(s) < auto_squeeze_min_lines:
                return write(s, tags)
            # Now the full line-count check.
            numoflines = self.count_lines(s)
            if numoflines < auto_squeeze_min_lines:
                return write(s, tags)

            # Create an ExpandingButton instance.
            expandingbutton = ExpandingButton(s, tags, numoflines, self)

            # Insert the ExpandingButton into the Text widget.
            text.mark_gravity("iomark", tk.RIGHT)
            text.window_create("iomark", window=expandingbutton,
                               padx=3, pady=5)
            text.see("iomark")
            text.update()
            text.mark_gravity("iomark", tk.LEFT)

            # Add the ExpandingButton to the Squeezer's list.
            self.expandingbuttons.append(expandingbutton)

        editwin.write = mywrite

    def count_lines(self, s):
        """Count the number of lines in a given text.

        Before calculation, the tab width and line length of the text are
        fetched, so that up-to-date values are used.

        Lines are counted as if the string was wrapped so that lines are never
        over linewidth characters long.

        Tabs are considered tabwidth characters long.
        """
        return count_lines_with_wrapping(s, self.editwin.width)

    def squeeze_current_text(self):
        """Squeeze the text block where the insertion cursor is.

        If the cursor is not in a squeezable block of text, give the
        user a small warning and do nothing.
        """
        # Set tag_name to the first valid tag found on the "insert" cursor.
        tag_names = self.text.tag_names(tk.INSERT)
        for tag_name in ("stdout", "stderr"):
            if tag_name in tag_names:
                break
        else:
            # The insert cursor doesn't have a "stdout" or "stderr" tag.
            self.text.bell()
            return "break"

        # Find the range to squeeze.
        start, end = self.text.tag_prevrange(tag_name, tk.INSERT + "+1c")
        s = self.text.get(start, end)

        # If the last char is a newline, remove it from the range.
        if len(s) > 0 and s[-1] == '\n':
            end = self.text.index("%s-1c" % end)
            s = s[:-1]

        # Delete the text.
        self.base_text.delete(start, end)

        # Prepare an ExpandingButton.
        numoflines = self.count_lines(s)
        expandingbutton = ExpandingButton(s, tag_name, numoflines, self)

        # insert the ExpandingButton to the Text
        self.text.window_create(start, window=expandingbutton,
                                padx=3, pady=5)

        # Insert the ExpandingButton to the list of ExpandingButtons,
        # while keeping the list ordered according to the position of
        # the buttons in the Text widget.
        i = len(self.expandingbuttons)
        while i > 0 and self.text.compare(self.expandingbuttons[i-1],
                                          ">", expandingbutton):
            i -= 1
        self.expandingbuttons.insert(i, expandingbutton)

        return "break"


Squeezer.reload()


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_squeezer', verbosity=2, exit=False)

    # Add htest.
