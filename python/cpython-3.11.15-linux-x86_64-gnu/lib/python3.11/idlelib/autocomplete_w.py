"""
An auto-completion window for IDLE, used by the autocomplete extension
"""
import platform

from tkinter import *
from tkinter.ttk import Scrollbar

from idlelib.autocomplete import FILES, ATTRS
from idlelib.multicall import MC_SHIFT

HIDE_VIRTUAL_EVENT_NAME = "<<autocompletewindow-hide>>"
HIDE_FOCUS_OUT_SEQUENCE = "<FocusOut>"
HIDE_SEQUENCES = (HIDE_FOCUS_OUT_SEQUENCE, "<ButtonPress>")
KEYPRESS_VIRTUAL_EVENT_NAME = "<<autocompletewindow-keypress>>"
# We need to bind event beyond <Key> so that the function will be called
# before the default specific IDLE function
KEYPRESS_SEQUENCES = ("<Key>", "<Key-BackSpace>", "<Key-Return>", "<Key-Tab>",
                      "<Key-Up>", "<Key-Down>", "<Key-Home>", "<Key-End>",
                      "<Key-Prior>", "<Key-Next>", "<Key-Escape>")
KEYRELEASE_VIRTUAL_EVENT_NAME = "<<autocompletewindow-keyrelease>>"
KEYRELEASE_SEQUENCE = "<KeyRelease>"
LISTUPDATE_SEQUENCE = "<B1-ButtonRelease>"
WINCONFIG_SEQUENCE = "<Configure>"
DOUBLECLICK_SEQUENCE = "<B1-Double-ButtonRelease>"

class AutoCompleteWindow:

    def __init__(self, widget, tags):
        # The widget (Text) on which we place the AutoCompleteWindow
        self.widget = widget
        # Tags to mark inserted text with
        self.tags = tags
        # The widgets we create
        self.autocompletewindow = self.listbox = self.scrollbar = None
        # The default foreground and background of a selection. Saved because
        # they are changed to the regular colors of list items when the
        # completion start is not a prefix of the selected completion
        self.origselforeground = self.origselbackground = None
        # The list of completions
        self.completions = None
        # A list with more completions, or None
        self.morecompletions = None
        # The completion mode, either autocomplete.ATTRS or .FILES.
        self.mode = None
        # The current completion start, on the text box (a string)
        self.start = None
        # The index of the start of the completion
        self.startindex = None
        # The last typed start, used so that when the selection changes,
        # the new start will be as close as possible to the last typed one.
        self.lasttypedstart = None
        # Do we have an indication that the user wants the completion window
        # (for example, he clicked the list)
        self.userwantswindow = None
        # event ids
        self.hideid = self.keypressid = self.listupdateid = \
            self.winconfigid = self.keyreleaseid = self.doubleclickid = None
        # Flag set if last keypress was a tab
        self.lastkey_was_tab = False
        # Flag set to avoid recursive <Configure> callback invocations.
        self.is_configuring = False

    def _change_start(self, newstart):
        min_len = min(len(self.start), len(newstart))
        i = 0
        while i < min_len and self.start[i] == newstart[i]:
            i += 1
        if i < len(self.start):
            self.widget.delete("%s+%dc" % (self.startindex, i),
                               "%s+%dc" % (self.startindex, len(self.start)))
        if i < len(newstart):
            self.widget.insert("%s+%dc" % (self.startindex, i),
                               newstart[i:],
                               self.tags)
        self.start = newstart

    def _binary_search(self, s):
        """Find the first index in self.completions where completions[i] is
        greater or equal to s, or the last index if there is no such.
        """
        i = 0; j = len(self.completions)
        while j > i:
            m = (i + j) // 2
            if self.completions[m] >= s:
                j = m
            else:
                i = m + 1
        return min(i, len(self.completions)-1)

    def _complete_string(self, s):
        """Assuming that s is the prefix of a string in self.completions,
        return the longest string which is a prefix of all the strings which
        s is a prefix of them. If s is not a prefix of a string, return s.
        """
        first = self._binary_search(s)
        if self.completions[first][:len(s)] != s:
            # There is not even one completion which s is a prefix of.
            return s
        # Find the end of the range of completions where s is a prefix of.
        i = first + 1
        j = len(self.completions)
        while j > i:
            m = (i + j) // 2
            if self.completions[m][:len(s)] != s:
                j = m
            else:
                i = m + 1
        last = i-1

        if first == last: # only one possible completion
            return self.completions[first]

        # We should return the maximum prefix of first and last
        first_comp = self.completions[first]
        last_comp = self.completions[last]
        min_len = min(len(first_comp), len(last_comp))
        i = len(s)
        while i < min_len and first_comp[i] == last_comp[i]:
            i += 1
        return first_comp[:i]

    def _selection_changed(self):
        """Call when the selection of the Listbox has changed.

        Updates the Listbox display and calls _change_start.
        """
        cursel = int(self.listbox.curselection()[0])

        self.listbox.see(cursel)

        lts = self.lasttypedstart
        selstart = self.completions[cursel]
        if self._binary_search(lts) == cursel:
            newstart = lts
        else:
            min_len = min(len(lts), len(selstart))
            i = 0
            while i < min_len and lts[i] == selstart[i]:
                i += 1
            newstart = selstart[:i]
        self._change_start(newstart)

        if self.completions[cursel][:len(self.start)] == self.start:
            # start is a prefix of the selected completion
            self.listbox.configure(selectbackground=self.origselbackground,
                                   selectforeground=self.origselforeground)
        else:
            self.listbox.configure(selectbackground=self.listbox.cget("bg"),
                                   selectforeground=self.listbox.cget("fg"))
            # If there are more completions, show them, and call me again.
            if self.morecompletions:
                self.completions = self.morecompletions
                self.morecompletions = None
                self.listbox.delete(0, END)
                for item in self.completions:
                    self.listbox.insert(END, item)
                self.listbox.select_set(self._binary_search(self.start))
                self._selection_changed()

    def show_window(self, comp_lists, index, complete, mode, userWantsWin):
        """Show the autocomplete list, bind events.

        If complete is True, complete the text, and if there is exactly
        one matching completion, don't open a list.
        """
        # Handle the start we already have
        self.completions, self.morecompletions = comp_lists
        self.mode = mode
        self.startindex = self.widget.index(index)
        self.start = self.widget.get(self.startindex, "insert")
        if complete:
            completed = self._complete_string(self.start)
            start = self.start
            self._change_start(completed)
            i = self._binary_search(completed)
            if self.completions[i] == completed and \
               (i == len(self.completions)-1 or
                self.completions[i+1][:len(completed)] != completed):
                # There is exactly one matching completion
                return completed == start
        self.userwantswindow = userWantsWin
        self.lasttypedstart = self.start

        self.autocompletewindow = acw = Toplevel(self.widget)
        acw.withdraw()
        acw.wm_overrideredirect(1)
        try:
            # Prevent grabbing focus on macOS.
            acw.tk.call("::tk::unsupported::MacWindowStyle", "style", acw._w,
                        "help", "noActivates")
        except TclError:
            pass
        self.scrollbar = scrollbar = Scrollbar(acw, orient=VERTICAL)
        self.listbox = listbox = Listbox(acw, yscrollcommand=scrollbar.set,
                                         exportselection=False)
        for item in self.completions:
            listbox.insert(END, item)
        self.origselforeground = listbox.cget("selectforeground")
        self.origselbackground = listbox.cget("selectbackground")
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        #acw.update_idletasks() # Need for tk8.6.8 on macOS: #40128.
        acw.lift()  # work around bug in Tk 8.5.18+ (issue #24570)

        # Initialize the listbox selection
        self.listbox.select_set(self._binary_search(self.start))
        self._selection_changed()

        # bind events
        self.hideaid = acw.bind(HIDE_VIRTUAL_EVENT_NAME, self.hide_event)
        self.hidewid = self.widget.bind(HIDE_VIRTUAL_EVENT_NAME, self.hide_event)
        acw.event_add(HIDE_VIRTUAL_EVENT_NAME, HIDE_FOCUS_OUT_SEQUENCE)
        for seq in HIDE_SEQUENCES:
            self.widget.event_add(HIDE_VIRTUAL_EVENT_NAME, seq)

        self.keypressid = self.widget.bind(KEYPRESS_VIRTUAL_EVENT_NAME,
                                           self.keypress_event)
        for seq in KEYPRESS_SEQUENCES:
            self.widget.event_add(KEYPRESS_VIRTUAL_EVENT_NAME, seq)
        self.keyreleaseid = self.widget.bind(KEYRELEASE_VIRTUAL_EVENT_NAME,
                                             self.keyrelease_event)
        self.widget.event_add(KEYRELEASE_VIRTUAL_EVENT_NAME,KEYRELEASE_SEQUENCE)
        self.listupdateid = listbox.bind(LISTUPDATE_SEQUENCE,
                                         self.listselect_event)
        self.is_configuring = False
        self.winconfigid = acw.bind(WINCONFIG_SEQUENCE, self.winconfig_event)
        self.doubleclickid = listbox.bind(DOUBLECLICK_SEQUENCE,
                                          self.doubleclick_event)
        return None

    def winconfig_event(self, event):
        if self.is_configuring:
            # Avoid running on recursive <Configure> callback invocations.
            return

        self.is_configuring = True
        if not self.is_active():
            return

        # Since the <Configure> event may occur after the completion window is gone,
        # catch potential TclError exceptions when accessing acw.  See: bpo-41611.
        try:
            # Position the completion list window
            text = self.widget
            text.see(self.startindex)
            x, y, cx, cy = text.bbox(self.startindex)
            acw = self.autocompletewindow
            if platform.system().startswith('Windows'):
                # On Windows an update() call is needed for the completion
                # list window to be created, so that we can fetch its width
                # and height.  However, this is not needed on other platforms
                # (tested on Ubuntu and macOS) but at one point began
                # causing freezes on macOS.  See issues 37849 and 41611.
                acw.update()
            acw_width, acw_height = acw.winfo_width(), acw.winfo_height()
            text_width, text_height = text.winfo_width(), text.winfo_height()
            new_x = text.winfo_rootx() + min(x, max(0, text_width - acw_width))
            new_y = text.winfo_rooty() + y
            if (text_height - (y + cy) >= acw_height # enough height below
                or y < acw_height): # not enough height above
                # place acw below current line
                new_y += cy
            else:
                # place acw above current line
                new_y -= acw_height
            acw.wm_geometry("+%d+%d" % (new_x, new_y))
            acw.deiconify()
            acw.update_idletasks()
        except TclError:
            pass

        if platform.system().startswith('Windows'):
            # See issue 15786.  When on Windows platform, Tk will misbehave
            # to call winconfig_event multiple times, we need to prevent this,
            # otherwise mouse button double click will not be able to used.
            try:
                acw.unbind(WINCONFIG_SEQUENCE, self.winconfigid)
            except TclError:
                pass
            self.winconfigid = None

        self.is_configuring = False

    def _hide_event_check(self):
        if not self.autocompletewindow:
            return

        try:
            if not self.autocompletewindow.focus_get():
                self.hide_window()
        except KeyError:
            # See issue 734176, when user click on menu, acw.focus_get()
            # will get KeyError.
            self.hide_window()

    def hide_event(self, event):
        # Hide autocomplete list if it exists and does not have focus or
        # mouse click on widget / text area.
        if self.is_active():
            if event.type == EventType.FocusOut:
                # On Windows platform, it will need to delay the check for
                # acw.focus_get() when click on acw, otherwise it will return
                # None and close the window
                self.widget.after(1, self._hide_event_check)
            elif event.type == EventType.ButtonPress:
                # ButtonPress event only bind to self.widget
                self.hide_window()

    def listselect_event(self, event):
        if self.is_active():
            self.userwantswindow = True
            cursel = int(self.listbox.curselection()[0])
            self._change_start(self.completions[cursel])

    def doubleclick_event(self, event):
        # Put the selected completion in the text, and close the list
        cursel = int(self.listbox.curselection()[0])
        self._change_start(self.completions[cursel])
        self.hide_window()

    def keypress_event(self, event):
        if not self.is_active():
            return None
        keysym = event.keysym
        if hasattr(event, "mc_state"):
            state = event.mc_state
        else:
            state = 0
        if keysym != "Tab":
            self.lastkey_was_tab = False
        if (len(keysym) == 1 or keysym in ("underscore", "BackSpace")
            or (self.mode == FILES and keysym in
                ("period", "minus"))) \
           and not (state & ~MC_SHIFT):
            # Normal editing of text
            if len(keysym) == 1:
                self._change_start(self.start + keysym)
            elif keysym == "underscore":
                self._change_start(self.start + '_')
            elif keysym == "period":
                self._change_start(self.start + '.')
            elif keysym == "minus":
                self._change_start(self.start + '-')
            else:
                # keysym == "BackSpace"
                if len(self.start) == 0:
                    self.hide_window()
                    return None
                self._change_start(self.start[:-1])
            self.lasttypedstart = self.start
            self.listbox.select_clear(0, int(self.listbox.curselection()[0]))
            self.listbox.select_set(self._binary_search(self.start))
            self._selection_changed()
            return "break"

        elif keysym == "Return":
            self.complete()
            self.hide_window()
            return 'break'

        elif (self.mode == ATTRS and keysym in
              ("period", "space", "parenleft", "parenright", "bracketleft",
               "bracketright")) or \
             (self.mode == FILES and keysym in
              ("slash", "backslash", "quotedbl", "apostrophe")) \
             and not (state & ~MC_SHIFT):
            # If start is a prefix of the selection, but is not '' when
            # completing file names, put the whole
            # selected completion. Anyway, close the list.
            cursel = int(self.listbox.curselection()[0])
            if self.completions[cursel][:len(self.start)] == self.start \
               and (self.mode == ATTRS or self.start):
                self._change_start(self.completions[cursel])
            self.hide_window()
            return None

        elif keysym in ("Home", "End", "Prior", "Next", "Up", "Down") and \
             not state:
            # Move the selection in the listbox
            self.userwantswindow = True
            cursel = int(self.listbox.curselection()[0])
            if keysym == "Home":
                newsel = 0
            elif keysym == "End":
                newsel = len(self.completions)-1
            elif keysym in ("Prior", "Next"):
                jump = self.listbox.nearest(self.listbox.winfo_height()) - \
                       self.listbox.nearest(0)
                if keysym == "Prior":
                    newsel = max(0, cursel-jump)
                else:
                    assert keysym == "Next"
                    newsel = min(len(self.completions)-1, cursel+jump)
            elif keysym == "Up":
                newsel = max(0, cursel-1)
            else:
                assert keysym == "Down"
                newsel = min(len(self.completions)-1, cursel+1)
            self.listbox.select_clear(cursel)
            self.listbox.select_set(newsel)
            self._selection_changed()
            self._change_start(self.completions[newsel])
            return "break"

        elif (keysym == "Tab" and not state):
            if self.lastkey_was_tab:
                # two tabs in a row; insert current selection and close acw
                cursel = int(self.listbox.curselection()[0])
                self._change_start(self.completions[cursel])
                self.hide_window()
                return "break"
            else:
                # first tab; let AutoComplete handle the completion
                self.userwantswindow = True
                self.lastkey_was_tab = True
                return None

        elif any(s in keysym for s in ("Shift", "Control", "Alt",
                                       "Meta", "Command", "Option")):
            # A modifier key, so ignore
            return None

        elif event.char and event.char >= ' ':
            # Regular character with a non-length-1 keycode
            self._change_start(self.start + event.char)
            self.lasttypedstart = self.start
            self.listbox.select_clear(0, int(self.listbox.curselection()[0]))
            self.listbox.select_set(self._binary_search(self.start))
            self._selection_changed()
            return "break"

        else:
            # Unknown event, close the window and let it through.
            self.hide_window()
            return None

    def keyrelease_event(self, event):
        if not self.is_active():
            return
        if self.widget.index("insert") != \
           self.widget.index("%s+%dc" % (self.startindex, len(self.start))):
            # If we didn't catch an event which moved the insert, close window
            self.hide_window()

    def is_active(self):
        return self.autocompletewindow is not None

    def complete(self):
        self._change_start(self._complete_string(self.start))
        # The selection doesn't change.

    def hide_window(self):
        if not self.is_active():
            return

        # unbind events
        self.autocompletewindow.event_delete(HIDE_VIRTUAL_EVENT_NAME,
                                             HIDE_FOCUS_OUT_SEQUENCE)
        for seq in HIDE_SEQUENCES:
            self.widget.event_delete(HIDE_VIRTUAL_EVENT_NAME, seq)

        self.autocompletewindow.unbind(HIDE_VIRTUAL_EVENT_NAME, self.hideaid)
        self.widget.unbind(HIDE_VIRTUAL_EVENT_NAME, self.hidewid)
        self.hideaid = None
        self.hidewid = None
        for seq in KEYPRESS_SEQUENCES:
            self.widget.event_delete(KEYPRESS_VIRTUAL_EVENT_NAME, seq)
        self.widget.unbind(KEYPRESS_VIRTUAL_EVENT_NAME, self.keypressid)
        self.keypressid = None
        self.widget.event_delete(KEYRELEASE_VIRTUAL_EVENT_NAME,
                                 KEYRELEASE_SEQUENCE)
        self.widget.unbind(KEYRELEASE_VIRTUAL_EVENT_NAME, self.keyreleaseid)
        self.keyreleaseid = None
        self.listbox.unbind(LISTUPDATE_SEQUENCE, self.listupdateid)
        self.listupdateid = None
        if self.winconfigid:
            self.autocompletewindow.unbind(WINCONFIG_SEQUENCE, self.winconfigid)
            self.winconfigid = None

        # Re-focusOn frame.text (See issue #15786)
        self.widget.focus_set()

        # destroy widgets
        self.scrollbar.destroy()
        self.scrollbar = None
        self.listbox.destroy()
        self.listbox = None
        self.autocompletewindow.destroy()
        self.autocompletewindow = None


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_autocomplete_w', verbosity=2, exit=False)

# TODO: autocomplete/w htest here
