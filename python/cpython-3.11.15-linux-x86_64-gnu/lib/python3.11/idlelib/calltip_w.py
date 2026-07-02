"""A call-tip window class for Tkinter/IDLE.

After tooltip.py, which uses ideas gleaned from PySol.
Used by calltip.py.
"""
from tkinter import Label, LEFT, SOLID, TclError

from idlelib.tooltip import TooltipBase

HIDE_EVENT = "<<calltipwindow-hide>>"
HIDE_SEQUENCES = ("<Key-Escape>", "<FocusOut>")
CHECKHIDE_EVENT = "<<calltipwindow-checkhide>>"
CHECKHIDE_SEQUENCES = ("<KeyRelease>", "<ButtonRelease>")
CHECKHIDE_TIME = 100  # milliseconds

MARK_RIGHT = "calltipwindowregion_right"


class CalltipWindow(TooltipBase):
    """A call-tip widget for tkinter text widgets."""

    def __init__(self, text_widget):
        """Create a call-tip; shown by showtip().

        text_widget: a Text widget with code for which call-tips are desired
        """
        # Note: The Text widget will be accessible as self.anchor_widget
        super().__init__(text_widget)

        self.label = self.text = None
        self.parenline = self.parencol = self.lastline = None
        self.hideid = self.checkhideid = None
        self.checkhide_after_id = None

    def get_position(self):
        """Choose the position of the call-tip."""
        curline = int(self.anchor_widget.index("insert").split('.')[0])
        if curline == self.parenline:
            anchor_index = (self.parenline, self.parencol)
        else:
            anchor_index = (curline, 0)
        box = self.anchor_widget.bbox("%d.%d" % anchor_index)
        if not box:
            box = list(self.anchor_widget.bbox("insert"))
            # align to left of window
            box[0] = 0
            box[2] = 0
        return box[0] + 2, box[1] + box[3]

    def position_window(self):
        "Reposition the window if needed."
        curline = int(self.anchor_widget.index("insert").split('.')[0])
        if curline == self.lastline:
            return
        self.lastline = curline
        self.anchor_widget.see("insert")
        super().position_window()

    def showtip(self, text, parenleft, parenright):
        """Show the call-tip, bind events which will close it and reposition it.

        text: the text to display in the call-tip
        parenleft: index of the opening parenthesis in the text widget
        parenright: index of the closing parenthesis in the text widget,
                    or the end of the line if there is no closing parenthesis
        """
        # Only called in calltip.Calltip, where lines are truncated
        self.text = text
        if self.tipwindow or not self.text:
            return

        self.anchor_widget.mark_set(MARK_RIGHT, parenright)
        self.parenline, self.parencol = map(
            int, self.anchor_widget.index(parenleft).split("."))

        super().showtip()

        self._bind_events()

    def showcontents(self):
        """Create the call-tip widget."""
        self.label = Label(self.tipwindow, text=self.text, justify=LEFT,
                           background="#ffffd0", foreground="black",
                           relief=SOLID, borderwidth=1,
                           font=self.anchor_widget['font'])
        self.label.pack()

    def checkhide_event(self, event=None):
        """Handle CHECK_HIDE_EVENT: call hidetip or reschedule."""
        if not self.tipwindow:
            # If the event was triggered by the same event that unbound
            # this function, the function will be called nevertheless,
            # so do nothing in this case.
            return None

        # Hide the call-tip if the insertion cursor moves outside of the
        # parenthesis.
        curline, curcol = map(int, self.anchor_widget.index("insert").split('.'))
        if curline < self.parenline or \
           (curline == self.parenline and curcol <= self.parencol) or \
           self.anchor_widget.compare("insert", ">", MARK_RIGHT):
            self.hidetip()
            return "break"

        # Not hiding the call-tip.

        self.position_window()
        # Re-schedule this function to be called again in a short while.
        if self.checkhide_after_id is not None:
            self.anchor_widget.after_cancel(self.checkhide_after_id)
        self.checkhide_after_id = \
            self.anchor_widget.after(CHECKHIDE_TIME, self.checkhide_event)
        return None

    def hide_event(self, event):
        """Handle HIDE_EVENT by calling hidetip."""
        if not self.tipwindow:
            # See the explanation in checkhide_event.
            return None
        self.hidetip()
        return "break"

    def hidetip(self):
        """Hide the call-tip."""
        if not self.tipwindow:
            return

        try:
            self.label.destroy()
        except TclError:
            pass
        self.label = None

        self.parenline = self.parencol = self.lastline = None
        try:
            self.anchor_widget.mark_unset(MARK_RIGHT)
        except TclError:
            pass

        try:
            self._unbind_events()
        except (TclError, ValueError):
            # ValueError may be raised by MultiCall
            pass

        super().hidetip()

    def _bind_events(self):
        """Bind event handlers."""
        self.checkhideid = self.anchor_widget.bind(CHECKHIDE_EVENT,
                                                   self.checkhide_event)
        for seq in CHECKHIDE_SEQUENCES:
            self.anchor_widget.event_add(CHECKHIDE_EVENT, seq)
        self.anchor_widget.after(CHECKHIDE_TIME, self.checkhide_event)
        self.hideid = self.anchor_widget.bind(HIDE_EVENT,
                                              self.hide_event)
        for seq in HIDE_SEQUENCES:
            self.anchor_widget.event_add(HIDE_EVENT, seq)

    def _unbind_events(self):
        """Unbind event handlers."""
        for seq in CHECKHIDE_SEQUENCES:
            self.anchor_widget.event_delete(CHECKHIDE_EVENT, seq)
        self.anchor_widget.unbind(CHECKHIDE_EVENT, self.checkhideid)
        self.checkhideid = None
        for seq in HIDE_SEQUENCES:
            self.anchor_widget.event_delete(HIDE_EVENT, seq)
        self.anchor_widget.unbind(HIDE_EVENT, self.hideid)
        self.hideid = None


def _calltip_window(parent):  # htest #
    from tkinter import Toplevel, Text, LEFT, BOTH

    top = Toplevel(parent)
    top.title("Test call-tips")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry("250x100+%d+%d" % (x + 175, y + 150))
    text = Text(top)
    text.pack(side=LEFT, fill=BOTH, expand=1)
    text.insert("insert", "string.split")
    top.update()

    calltip = CalltipWindow(text)
    def calltip_show(event):
        calltip.showtip("(s='Hello world')", "insert", "end")
    def calltip_hide(event):
        calltip.hidetip()
    text.event_add("<<calltip-show>>", "(")
    text.event_add("<<calltip-hide>>", ")")
    text.bind("<<calltip-show>>", calltip_show)
    text.bind("<<calltip-hide>>", calltip_hide)

    text.focus_set()


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_calltip_w', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_calltip_window)
