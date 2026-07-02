"""ParenMatch -- for parenthesis matching.

When you hit a right paren, the cursor should move briefly to the left
paren.  Paren here is used generically; the matching applies to
parentheses, square brackets, and curly braces.
"""
from idlelib.hyperparser import HyperParser
from idlelib.config import idleConf

_openers = {')':'(',']':'[','}':'{'}
CHECK_DELAY = 100 # milliseconds

class ParenMatch:
    """Highlight matching openers and closers, (), [], and {}.

    There are three supported styles of paren matching.  When a right
    paren (opener) is typed:

    opener -- highlight the matching left paren (closer);
    parens -- highlight the left and right parens (opener and closer);
    expression -- highlight the entire expression from opener to closer.
    (For back compatibility, 'default' is a synonym for 'opener').

    Flash-delay is the maximum milliseconds the highlighting remains.
    Any cursor movement (key press or click) before that removes the
    highlight.  If flash-delay is 0, there is no maximum.

    TODO:
    - Augment bell() with mismatch warning in status window.
    - Highlight when cursor is moved to the right of a closer.
      This might be too expensive to check.
    """

    RESTORE_VIRTUAL_EVENT_NAME = "<<parenmatch-check-restore>>"
    # We want the restore event be called before the usual return and
    # backspace events.
    RESTORE_SEQUENCES = ("<KeyPress>", "<ButtonPress>",
                         "<Key-Return>", "<Key-BackSpace>")

    def __init__(self, editwin):
        self.editwin = editwin
        self.text = editwin.text
        # Bind the check-restore event to the function restore_event,
        # so that we can then use activate_restore (which calls event_add)
        # and deactivate_restore (which calls event_delete).
        editwin.text.bind(self.RESTORE_VIRTUAL_EVENT_NAME,
                          self.restore_event)
        self.counter = 0
        self.is_restore_active = 0

    @classmethod
    def reload(cls):
        cls.STYLE = idleConf.GetOption(
            'extensions','ParenMatch','style', default='opener')
        cls.FLASH_DELAY = idleConf.GetOption(
                'extensions','ParenMatch','flash-delay', type='int',default=500)
        cls.BELL = idleConf.GetOption(
                'extensions','ParenMatch','bell', type='bool', default=1)
        cls.HILITE_CONFIG = idleConf.GetHighlight(idleConf.CurrentTheme(),
                                                  'hilite')

    def activate_restore(self):
        "Activate mechanism to restore text from highlighting."
        if not self.is_restore_active:
            for seq in self.RESTORE_SEQUENCES:
                self.text.event_add(self.RESTORE_VIRTUAL_EVENT_NAME, seq)
            self.is_restore_active = True

    def deactivate_restore(self):
        "Remove restore event bindings."
        if self.is_restore_active:
            for seq in self.RESTORE_SEQUENCES:
                self.text.event_delete(self.RESTORE_VIRTUAL_EVENT_NAME, seq)
            self.is_restore_active = False

    def flash_paren_event(self, event):
        "Handle editor 'show surrounding parens' event (menu or shortcut)."
        indices = (HyperParser(self.editwin, "insert")
                   .get_surrounding_brackets())
        self.finish_paren_event(indices)
        return "break"

    def paren_closed_event(self, event):
        "Handle user input of closer."
        # If user bound non-closer to <<paren-closed>>, quit.
        closer = self.text.get("insert-1c")
        if closer not in _openers:
            return
        hp = HyperParser(self.editwin, "insert-1c")
        if not hp.is_in_code():
            return
        indices = hp.get_surrounding_brackets(_openers[closer], True)
        self.finish_paren_event(indices)
        return  # Allow calltips to see ')'

    def finish_paren_event(self, indices):
        if indices is None and self.BELL:
            self.text.bell()
            return
        self.activate_restore()
        # self.create_tag(indices)
        self.tagfuncs.get(self.STYLE, self.create_tag_expression)(self, indices)
        # self.set_timeout()
        (self.set_timeout_last if self.FLASH_DELAY else
                            self.set_timeout_none)()

    def restore_event(self, event=None):
        "Remove effect of doing match."
        self.text.tag_delete("paren")
        self.deactivate_restore()
        self.counter += 1   # disable the last timer, if there is one.

    def handle_restore_timer(self, timer_count):
        if timer_count == self.counter:
            self.restore_event()

    # any one of the create_tag_XXX methods can be used depending on
    # the style

    def create_tag_opener(self, indices):
        """Highlight the single paren that matches"""
        self.text.tag_add("paren", indices[0])
        self.text.tag_config("paren", self.HILITE_CONFIG)

    def create_tag_parens(self, indices):
        """Highlight the left and right parens"""
        if self.text.get(indices[1]) in (')', ']', '}'):
            rightindex = indices[1]+"+1c"
        else:
            rightindex = indices[1]
        self.text.tag_add("paren", indices[0], indices[0]+"+1c", rightindex+"-1c", rightindex)
        self.text.tag_config("paren", self.HILITE_CONFIG)

    def create_tag_expression(self, indices):
        """Highlight the entire expression"""
        if self.text.get(indices[1]) in (')', ']', '}'):
            rightindex = indices[1]+"+1c"
        else:
            rightindex = indices[1]
        self.text.tag_add("paren", indices[0], rightindex)
        self.text.tag_config("paren", self.HILITE_CONFIG)

    tagfuncs = {
        'opener': create_tag_opener,
        'default': create_tag_opener,
        'parens': create_tag_parens,
        'expression': create_tag_expression,
        }

    # any one of the set_timeout_XXX methods can be used depending on
    # the style

    def set_timeout_none(self):
        """Highlight will remain until user input turns it off
        or the insert has moved"""
        # After CHECK_DELAY, call a function which disables the "paren" tag
        # if the event is for the most recent timer and the insert has changed,
        # or schedules another call for itself.
        self.counter += 1
        def callme(callme, self=self, c=self.counter,
                   index=self.text.index("insert")):
            if index != self.text.index("insert"):
                self.handle_restore_timer(c)
            else:
                self.editwin.text_frame.after(CHECK_DELAY, callme, callme)
        self.editwin.text_frame.after(CHECK_DELAY, callme, callme)

    def set_timeout_last(self):
        """The last highlight created will be removed after FLASH_DELAY millisecs"""
        # associate a counter with an event; only disable the "paren"
        # tag if the event is for the most recent timer.
        self.counter += 1
        self.editwin.text_frame.after(
            self.FLASH_DELAY,
            lambda self=self, c=self.counter: self.handle_restore_timer(c))


ParenMatch.reload()


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_parenmatch', verbosity=2)
