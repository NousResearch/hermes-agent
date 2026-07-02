"""
Dialog for building Tkinter accelerator key bindings
"""
from tkinter import Toplevel, Listbox, StringVar, TclError
from tkinter.ttk import Frame, Button, Checkbutton, Entry, Label, Scrollbar
from tkinter import messagebox
from tkinter.simpledialog import _setup_dialog
import string
import sys


FUNCTION_KEYS = ('F1', 'F2' ,'F3' ,'F4' ,'F5' ,'F6',
                 'F7', 'F8' ,'F9' ,'F10' ,'F11' ,'F12')
ALPHANUM_KEYS = tuple(string.ascii_lowercase + string.digits)
PUNCTUATION_KEYS = tuple('~!@#%^&*()_-+={}[]|;:,.<>/?')
WHITESPACE_KEYS = ('Tab', 'Space', 'Return')
EDIT_KEYS = ('BackSpace', 'Delete', 'Insert')
MOVE_KEYS = ('Home', 'End', 'Page Up', 'Page Down', 'Left Arrow',
             'Right Arrow', 'Up Arrow', 'Down Arrow')
AVAILABLE_KEYS = (ALPHANUM_KEYS + PUNCTUATION_KEYS + FUNCTION_KEYS +
                  WHITESPACE_KEYS + EDIT_KEYS + MOVE_KEYS)


def translate_key(key, modifiers):
    "Translate from keycap symbol to the Tkinter keysym."
    mapping = {'Space':'space',
            '~':'asciitilde', '!':'exclam', '@':'at', '#':'numbersign',
            '%':'percent', '^':'asciicircum', '&':'ampersand',
            '*':'asterisk', '(':'parenleft', ')':'parenright',
            '_':'underscore', '-':'minus', '+':'plus', '=':'equal',
            '{':'braceleft', '}':'braceright',
            '[':'bracketleft', ']':'bracketright', '|':'bar',
            ';':'semicolon', ':':'colon', ',':'comma', '.':'period',
            '<':'less', '>':'greater', '/':'slash', '?':'question',
            'Page Up':'Prior', 'Page Down':'Next',
            'Left Arrow':'Left', 'Right Arrow':'Right',
            'Up Arrow':'Up', 'Down Arrow': 'Down', 'Tab':'Tab'}
    key = mapping.get(key, key)
    if 'Shift' in modifiers and key in string.ascii_lowercase:
        key = key.upper()
    return f'Key-{key}'


class GetKeysFrame(Frame):

    # Dialog title for invalid key sequence
    keyerror_title = 'Key Sequence Error'

    def __init__(self, parent, action, current_key_sequences):
        """
        parent - parent of this dialog
        action - the name of the virtual event these keys will be
                 mapped to
        current_key_sequences - a list of all key sequence lists
                 currently mapped to virtual events, for overlap checking
        """
        super().__init__(parent)
        self['borderwidth'] = 2
        self['relief'] = 'sunken'
        self.parent = parent
        self.action = action
        self.current_key_sequences = current_key_sequences
        self.result = ''
        self.key_string = StringVar(self)
        self.key_string.set('')
        # Set self.modifiers, self.modifier_label.
        self.set_modifiers_for_platform()
        self.modifier_vars = []
        for modifier in self.modifiers:
            variable = StringVar(self)
            variable.set('')
            self.modifier_vars.append(variable)
        self.advanced = False
        self.create_widgets()

    def showerror(self, *args, **kwargs):
        # Make testing easier.  Replace in #30751.
        messagebox.showerror(*args, **kwargs)

    def create_widgets(self):
        # Basic entry key sequence.
        self.frame_keyseq_basic = Frame(self, name='keyseq_basic')
        self.frame_keyseq_basic.grid(row=0, column=0, sticky='nsew',
                                      padx=5, pady=5)
        basic_title = Label(self.frame_keyseq_basic,
                            text=f"New keys for '{self.action}' :")
        basic_title.pack(anchor='w')

        basic_keys = Label(self.frame_keyseq_basic, justify='left',
                           textvariable=self.key_string, relief='groove',
                           borderwidth=2)
        basic_keys.pack(ipadx=5, ipady=5, fill='x')

        # Basic entry controls.
        self.frame_controls_basic = Frame(self)
        self.frame_controls_basic.grid(row=1, column=0, sticky='nsew', padx=5)

        # Basic entry modifiers.
        self.modifier_checkbuttons = {}
        column = 0
        for modifier, variable in zip(self.modifiers, self.modifier_vars):
            label = self.modifier_label.get(modifier, modifier)
            check = Checkbutton(self.frame_controls_basic,
                                command=self.build_key_string, text=label,
                                variable=variable, onvalue=modifier, offvalue='')
            check.grid(row=0, column=column, padx=2, sticky='w')
            self.modifier_checkbuttons[modifier] = check
            column += 1

        # Basic entry help text.
        help_basic = Label(self.frame_controls_basic, justify='left',
                           text="Select the desired modifier keys\n"+
                                "above, and the final key from the\n"+
                                "list on the right.\n\n" +
                                "Use upper case Symbols when using\n" +
                                "the Shift modifier.  (Letters will be\n" +
                                "converted automatically.)")
        help_basic.grid(row=1, column=0, columnspan=4, padx=2, sticky='w')

        # Basic entry key list.
        self.list_keys_final = Listbox(self.frame_controls_basic, width=15,
                                       height=10, selectmode='single')
        self.list_keys_final.insert('end', *AVAILABLE_KEYS)
        self.list_keys_final.bind('<ButtonRelease-1>', self.final_key_selected)
        self.list_keys_final.grid(row=0, column=4, rowspan=4, sticky='ns')
        scroll_keys_final = Scrollbar(self.frame_controls_basic,
                                      orient='vertical',
                                      command=self.list_keys_final.yview)
        self.list_keys_final.config(yscrollcommand=scroll_keys_final.set)
        scroll_keys_final.grid(row=0, column=5, rowspan=4, sticky='ns')
        self.button_clear = Button(self.frame_controls_basic,
                                   text='Clear Keys',
                                   command=self.clear_key_seq)
        self.button_clear.grid(row=2, column=0, columnspan=4)

        # Advanced entry key sequence.
        self.frame_keyseq_advanced = Frame(self, name='keyseq_advanced')
        self.frame_keyseq_advanced.grid(row=0, column=0, sticky='nsew',
                                         padx=5, pady=5)
        advanced_title = Label(self.frame_keyseq_advanced, justify='left',
                               text=f"Enter new binding(s) for '{self.action}' :\n" +
                                     "(These bindings will not be checked for validity!)")
        advanced_title.pack(anchor='w')
        self.advanced_keys = Entry(self.frame_keyseq_advanced,
                                   textvariable=self.key_string)
        self.advanced_keys.pack(fill='x')

        # Advanced entry help text.
        self.frame_help_advanced = Frame(self)
        self.frame_help_advanced.grid(row=1, column=0, sticky='nsew', padx=5)
        help_advanced = Label(self.frame_help_advanced, justify='left',
            text="Key bindings are specified using Tkinter keysyms as\n"+
                 "in these samples: <Control-f>, <Shift-F2>, <F12>,\n"
                 "<Control-space>, <Meta-less>, <Control-Alt-Shift-X>.\n"
                 "Upper case is used when the Shift modifier is present!\n\n" +
                 "'Emacs style' multi-keystroke bindings are specified as\n" +
                 "follows: <Control-x><Control-y>, where the first key\n" +
                 "is the 'do-nothing' keybinding.\n\n" +
                 "Multiple separate bindings for one action should be\n"+
                 "separated by a space, eg., <Alt-v> <Meta-v>." )
        help_advanced.grid(row=0, column=0, sticky='nsew')

        # Switch between basic and advanced.
        self.button_level = Button(self, command=self.toggle_level,
                                  text='<< Basic Key Binding Entry')
        self.button_level.grid(row=2, column=0, stick='ew', padx=5, pady=5)
        self.toggle_level()

    def set_modifiers_for_platform(self):
        """Determine list of names of key modifiers for this platform.

        The names are used to build Tk bindings -- it doesn't matter if the
        keyboard has these keys; it matters if Tk understands them.  The
        order is also important: key binding equality depends on it, so
        config-keys.def must use the same ordering.
        """
        if sys.platform == "darwin":
            self.modifiers = ['Shift', 'Control', 'Option', 'Command']
        else:
            self.modifiers = ['Control', 'Alt', 'Shift']
        self.modifier_label = {'Control': 'Ctrl'}  # Short name.

    def toggle_level(self):
        "Toggle between basic and advanced keys."
        if  self.button_level.cget('text').startswith('Advanced'):
            self.clear_key_seq()
            self.button_level.config(text='<< Basic Key Binding Entry')
            self.frame_keyseq_advanced.lift()
            self.frame_help_advanced.lift()
            self.advanced_keys.focus_set()
            self.advanced = True
        else:
            self.clear_key_seq()
            self.button_level.config(text='Advanced Key Binding Entry >>')
            self.frame_keyseq_basic.lift()
            self.frame_controls_basic.lift()
            self.advanced = False

    def final_key_selected(self, event=None):
        "Handler for clicking on key in basic settings list."
        self.build_key_string()

    def build_key_string(self):
        "Create formatted string of modifiers plus the key."
        keylist = modifiers = self.get_modifiers()
        final_key = self.list_keys_final.get('anchor')
        if final_key:
            final_key = translate_key(final_key, modifiers)
            keylist.append(final_key)
        self.key_string.set(f"<{'-'.join(keylist)}>")

    def get_modifiers(self):
        "Return ordered list of modifiers that have been selected."
        mod_list = [variable.get() for variable in self.modifier_vars]
        return [mod for mod in mod_list if mod]

    def clear_key_seq(self):
        "Clear modifiers and keys selection."
        self.list_keys_final.select_clear(0, 'end')
        self.list_keys_final.yview('moveto', '0.0')
        for variable in self.modifier_vars:
            variable.set('')
        self.key_string.set('')

    def ok(self):
        self.result = ''
        keys = self.key_string.get().strip()
        if not keys:
            self.showerror(title=self.keyerror_title, parent=self,
                           message="No key specified.")
            return
        if (self.advanced or self.keys_ok(keys)) and self.bind_ok(keys):
            self.result = keys
        return

    def keys_ok(self, keys):
        """Validity check on user's 'basic' keybinding selection.

        Doesn't check the string produced by the advanced dialog because
        'modifiers' isn't set.
        """
        final_key = self.list_keys_final.get('anchor')
        modifiers = self.get_modifiers()
        title = self.keyerror_title
        key_sequences = [key for keylist in self.current_key_sequences
                             for key in keylist]
        if not keys.endswith('>'):
            self.showerror(title, parent=self,
                           message='Missing the final Key')
        elif (not modifiers
              and final_key not in FUNCTION_KEYS + MOVE_KEYS):
            self.showerror(title=title, parent=self,
                           message='No modifier key(s) specified.')
        elif (modifiers == ['Shift']) \
                 and (final_key not in
                      FUNCTION_KEYS + MOVE_KEYS + ('Tab', 'Space')):
            msg = 'The shift modifier by itself may not be used with'\
                  ' this key symbol.'
            self.showerror(title=title, parent=self, message=msg)
        elif keys in key_sequences:
            msg = 'This key combination is already in use.'
            self.showerror(title=title, parent=self, message=msg)
        else:
            return True
        return False

    def bind_ok(self, keys):
        "Return True if Tcl accepts the new keys else show message."
        try:
            binding = self.bind(keys, lambda: None)
        except TclError as err:
            self.showerror(
                    title=self.keyerror_title, parent=self,
                    message=(f'The entered key sequence is not accepted.\n\n'
                             f'Error: {err}'))
            return False
        else:
            self.unbind(keys, binding)
            return True


class GetKeysWindow(Toplevel):

    def __init__(self, parent, title, action, current_key_sequences,
                 *, _htest=False, _utest=False):
        """
        parent - parent of this dialog
        title - string which is the title of the popup dialog
        action - string, the name of the virtual event these keys will be
                 mapped to
        current_key_sequences - list, a list of all key sequence lists
                 currently mapped to virtual events, for overlap checking
        _htest - bool, change box location when running htest
        _utest - bool, do not wait when running unittest
        """
        super().__init__(parent)
        self.withdraw()  # Hide while setting geometry.
        self['borderwidth'] = 5
        self.resizable(height=False, width=False)
        # Needed for winfo_reqwidth().
        self.update_idletasks()
        # Center dialog over parent (or below htest box).
        x = (parent.winfo_rootx() +
             (parent.winfo_width()//2 - self.winfo_reqwidth()//2))
        y = (parent.winfo_rooty() +
             ((parent.winfo_height()//2 - self.winfo_reqheight()//2)
              if not _htest else 150))
        self.geometry(f"+{x}+{y}")

        self.title(title)
        self.frame = frame = GetKeysFrame(self, action, current_key_sequences)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        frame_buttons = Frame(self)
        self.button_ok = Button(frame_buttons, text='OK',
                                width=8, command=self.ok)
        self.button_cancel = Button(frame_buttons, text='Cancel',
                                   width=8, command=self.cancel)
        self.button_ok.grid(row=0, column=0, padx=5, pady=5)
        self.button_cancel.grid(row=0, column=1, padx=5, pady=5)
        frame.pack(side='top', expand=True, fill='both')
        frame_buttons.pack(side='bottom', fill='x')

        self.transient(parent)
        _setup_dialog(self)
        self.grab_set()
        if not _utest:
            self.deiconify()  # Geometry set, unhide.
            self.wait_window()

    @property
    def result(self):
        return self.frame.result

    @result.setter
    def result(self, value):
        self.frame.result = value

    def ok(self, event=None):
        self.frame.ok()
        self.grab_release()
        self.destroy()

    def cancel(self, event=None):
        self.result = ''
        self.grab_release()
        self.destroy()


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_config_key', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(GetKeysWindow)
