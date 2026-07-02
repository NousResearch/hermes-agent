"""Ttk wrapper.

This module provides classes to allow using Tk themed widget set.

Ttk is based on a revised and enhanced version of
TIP #48 (http://tip.tcl.tk/48) specified style engine.

Its basic idea is to separate, to the extent possible, the code
implementing a widget's behavior from the code implementing its
appearance. Widget class bindings are primarily responsible for
maintaining the widget state and invoking callbacks, all aspects
of the widgets appearance lies at Themes.
"""

__version__ = "0.3.1"

__author__ = "Guilherme Polo <ggpolo@gmail.com>"

__all__ = ["Button", "Checkbutton", "Combobox", "Entry", "Frame", "Label",
           "Labelframe", "LabelFrame", "Menubutton", "Notebook", "Panedwindow",
           "PanedWindow", "Progressbar", "Radiobutton", "Scale", "Scrollbar",
           "Separator", "Sizegrip", "Spinbox", "Style", "Treeview",
           # Extensions
           "LabeledScale", "OptionMenu",
           # functions
           "tclobjs_to_py", "setup_master"]

import tkinter
from tkinter import _flatten, _join, _stringify, _splitdict


def _format_optvalue(value, script=False):
    """Internal function."""
    if script:
        # if caller passes a Tcl script to tk.call, all the values need to
        # be grouped into words (arguments to a command in Tcl dialect)
        value = _stringify(value)
    elif isinstance(value, (list, tuple)):
        value = _join(value)
    return value

def _format_optdict(optdict, script=False, ignore=None):
    """Formats optdict to a tuple to pass it to tk.call.

    E.g. (script=False):
      {'foreground': 'blue', 'padding': [1, 2, 3, 4]} returns:
      ('-foreground', 'blue', '-padding', '1 2 3 4')"""

    opts = []
    for opt, value in optdict.items():
        if not ignore or opt not in ignore:
            opts.append("-%s" % opt)
            if value is not None:
                opts.append(_format_optvalue(value, script))

    return _flatten(opts)

def _mapdict_values(items):
    # each value in mapdict is expected to be a sequence, where each item
    # is another sequence containing a state (or several) and a value
    # E.g. (script=False):
    #   [('active', 'selected', 'grey'), ('focus', [1, 2, 3, 4])]
    #   returns:
    #   ['active selected', 'grey', 'focus', [1, 2, 3, 4]]
    opt_val = []
    for *state, val in items:
        if len(state) == 1:
            # if it is empty (something that evaluates to False), then
            # format it to Tcl code to denote the "normal" state
            state = state[0] or ''
        else:
            # group multiple states
            state = ' '.join(state) # raise TypeError if not str
        opt_val.append(state)
        if val is not None:
            opt_val.append(val)
    return opt_val

def _format_mapdict(mapdict, script=False):
    """Formats mapdict to pass it to tk.call.

    E.g. (script=False):
      {'expand': [('active', 'selected', 'grey'), ('focus', [1, 2, 3, 4])]}

      returns:

      ('-expand', '{active selected} grey focus {1, 2, 3, 4}')"""

    opts = []
    for opt, value in mapdict.items():
        opts.extend(("-%s" % opt,
                     _format_optvalue(_mapdict_values(value), script)))

    return _flatten(opts)

def _format_elemcreate(etype, script=False, *args, **kw):
    """Formats args and kw according to the given element factory etype."""
    spec = None
    opts = ()
    if etype in ("image", "vsapi"):
        if etype == "image": # define an element based on an image
            # first arg should be the default image name
            iname = args[0]
            # next args, if any, are statespec/value pairs which is almost
            # a mapdict, but we just need the value
            imagespec = _join(_mapdict_values(args[1:]))
            spec = "%s %s" % (iname, imagespec)

        else:
            # define an element whose visual appearance is drawn using the
            # Microsoft Visual Styles API which is responsible for the
            # themed styles on Windows XP and Vista.
            # Availability: Tk 8.6, Windows XP and Vista.
            class_name, part_id = args[:2]
            statemap = _join(_mapdict_values(args[2:]))
            spec = "%s %s %s" % (class_name, part_id, statemap)

        opts = _format_optdict(kw, script)

    elif etype == "from": # clone an element
        # it expects a themename and optionally an element to clone from,
        # otherwise it will clone {} (empty element)
        spec = args[0] # theme name
        if len(args) > 1: # elementfrom specified
            opts = (_format_optvalue(args[1], script),)

    if script:
        spec = '{%s}' % spec
        opts = ' '.join(opts)

    return spec, opts

def _format_layoutlist(layout, indent=0, indent_size=2):
    """Formats a layout list so we can pass the result to ttk::style
    layout and ttk::style settings. Note that the layout doesn't have to
    be a list necessarily.

    E.g.:
      [("Menubutton.background", None),
       ("Menubutton.button", {"children":
           [("Menubutton.focus", {"children":
               [("Menubutton.padding", {"children":
                [("Menubutton.label", {"side": "left", "expand": 1})]
               })]
           })]
       }),
       ("Menubutton.indicator", {"side": "right"})
      ]

      returns:

      Menubutton.background
      Menubutton.button -children {
        Menubutton.focus -children {
          Menubutton.padding -children {
            Menubutton.label -side left -expand 1
          }
        }
      }
      Menubutton.indicator -side right"""
    script = []

    for layout_elem in layout:
        elem, opts = layout_elem
        opts = opts or {}
        fopts = ' '.join(_format_optdict(opts, True, ("children",)))
        head = "%s%s%s" % (' ' * indent, elem, (" %s" % fopts) if fopts else '')

        if "children" in opts:
            script.append(head + " -children {")
            indent += indent_size
            newscript, indent = _format_layoutlist(opts['children'], indent,
                indent_size)
            script.append(newscript)
            indent -= indent_size
            script.append('%s}' % (' ' * indent))
        else:
            script.append(head)

    return '\n'.join(script), indent

def _script_from_settings(settings):
    """Returns an appropriate script, based on settings, according to
    theme_settings definition to be used by theme_settings and
    theme_create."""
    script = []
    # a script will be generated according to settings passed, which
    # will then be evaluated by Tcl
    for name, opts in settings.items():
        # will format specific keys according to Tcl code
        if opts.get('configure'): # format 'configure'
            s = ' '.join(_format_optdict(opts['configure'], True))
            script.append("ttk::style configure %s %s;" % (name, s))

        if opts.get('map'): # format 'map'
            s = ' '.join(_format_mapdict(opts['map'], True))
            script.append("ttk::style map %s %s;" % (name, s))

        if 'layout' in opts: # format 'layout' which may be empty
            if not opts['layout']:
                s = 'null' # could be any other word, but this one makes sense
            else:
                s, _ = _format_layoutlist(opts['layout'])
            script.append("ttk::style layout %s {\n%s\n}" % (name, s))

        if opts.get('element create'): # format 'element create'
            eopts = opts['element create']
            etype = eopts[0]

            # find where args end, and where kwargs start
            argc = 1 # etype was the first one
            while argc < len(eopts) and not hasattr(eopts[argc], 'items'):
                argc += 1

            elemargs = eopts[1:argc]
            elemkw = eopts[argc] if argc < len(eopts) and eopts[argc] else {}
            spec, opts = _format_elemcreate(etype, True, *elemargs, **elemkw)

            script.append("ttk::style element create %s %s %s %s" % (
                name, etype, spec, opts))

    return '\n'.join(script)

def _list_from_statespec(stuple):
    """Construct a list from the given statespec tuple according to the
    accepted statespec accepted by _format_mapdict."""
    if isinstance(stuple, str):
        return stuple
    result = []
    it = iter(stuple)
    for state, val in zip(it, it):
        if hasattr(state, 'typename'):  # this is a Tcl object
            state = str(state).split()
        elif isinstance(state, str):
            state = state.split()
        elif not isinstance(state, (tuple, list)):
            state = (state,)
        if hasattr(val, 'typename'):
            val = str(val)
        result.append((*state, val))

    return result

def _list_from_layouttuple(tk, ltuple):
    """Construct a list from the tuple returned by ttk::layout, this is
    somewhat the reverse of _format_layoutlist."""
    ltuple = tk.splitlist(ltuple)
    res = []

    indx = 0
    while indx < len(ltuple):
        name = ltuple[indx]
        opts = {}
        res.append((name, opts))
        indx += 1

        while indx < len(ltuple): # grab name's options
            opt, val = ltuple[indx:indx + 2]
            if not opt.startswith('-'): # found next name
                break

            opt = opt[1:] # remove the '-' from the option
            indx += 2

            if opt == 'children':
                val = _list_from_layouttuple(tk, val)

            opts[opt] = val

    return res

def _val_or_dict(tk, options, *args):
    """Format options then call Tk command with args and options and return
    the appropriate result.

    If no option is specified, a dict is returned. If an option is
    specified with the None value, the value for that option is returned.
    Otherwise, the function just sets the passed options and the caller
    shouldn't be expecting a return value anyway."""
    options = _format_optdict(options)
    res = tk.call(*(args + options))

    if len(options) % 2: # option specified without a value, return its value
        return res

    return _splitdict(tk, res, conv=_tclobj_to_py)

def _convert_stringval(value):
    """Converts a value to, hopefully, a more appropriate Python object."""
    value = str(value)
    try:
        value = int(value)
    except (ValueError, TypeError):
        pass

    return value

def _to_number(x):
    if isinstance(x, str):
        if '.' in x:
            x = float(x)
        else:
            x = int(x)
    return x

def _tclobj_to_py(val):
    """Return value converted from Tcl object to Python object."""
    if val and hasattr(val, '__len__') and not isinstance(val, str):
        if getattr(val[0], 'typename', None) == 'StateSpec':
            val = _list_from_statespec(val)
        else:
            val = list(map(_convert_stringval, val))

    elif hasattr(val, 'typename'): # some other (single) Tcl object
        val = _convert_stringval(val)

    return val

def tclobjs_to_py(adict):
    """Returns adict with its values converted from Tcl objects to Python
    objects."""
    for opt, val in adict.items():
        adict[opt] = _tclobj_to_py(val)

    return adict

def setup_master(master=None):
    """If master is not None, itself is returned. If master is None,
    the default master is returned if there is one, otherwise a new
    master is created and returned.

    If it is not allowed to use the default root and master is None,
    RuntimeError is raised."""
    if master is None:
        master = tkinter._get_default_root()
    return master


class Style(object):
    """Manipulate style database."""

    _name = "ttk::style"

    def __init__(self, master=None):
        master = setup_master(master)
        self.master = master
        self.tk = self.master.tk


    def configure(self, style, query_opt=None, **kw):
        """Query or sets the default value of the specified option(s) in
        style.

        Each key in kw is an option and each value is either a string or
        a sequence identifying the value for that option."""
        if query_opt is not None:
            kw[query_opt] = None
        result = _val_or_dict(self.tk, kw, self._name, "configure", style)
        if result or query_opt:
            return result


    def map(self, style, query_opt=None, **kw):
        """Query or sets dynamic values of the specified option(s) in
        style.

        Each key in kw is an option and each value should be a list or a
        tuple (usually) containing statespecs grouped in tuples, or list,
        or something else of your preference. A statespec is compound of
        one or more states and then a value."""
        if query_opt is not None:
            result = self.tk.call(self._name, "map", style, '-%s' % query_opt)
            return _list_from_statespec(self.tk.splitlist(result))

        result = self.tk.call(self._name, "map", style, *_format_mapdict(kw))
        return {k: _list_from_statespec(self.tk.splitlist(v))
                for k, v in _splitdict(self.tk, result).items()}


    def lookup(self, style, option, state=None, default=None):
        """Returns the value specified for option in style.

        If state is specified it is expected to be a sequence of one
        or more states. If the default argument is set, it is used as
        a fallback value in case no specification for option is found."""
        state = ' '.join(state) if state else ''

        return self.tk.call(self._name, "lookup", style, '-%s' % option,
            state, default)


    def layout(self, style, layoutspec=None):
        """Define the widget layout for given style. If layoutspec is
        omitted, return the layout specification for given style.

        layoutspec is expected to be a list or an object different than
        None that evaluates to False if you want to "turn off" that style.
        If it is a list (or tuple, or something else), each item should be
        a tuple where the first item is the layout name and the second item
        should have the format described below:

        LAYOUTS

            A layout can contain the value None, if takes no options, or
            a dict of options specifying how to arrange the element.
            The layout mechanism uses a simplified version of the pack
            geometry manager: given an initial cavity, each element is
            allocated a parcel. Valid options/values are:

                side: whichside
                    Specifies which side of the cavity to place the
                    element; one of top, right, bottom or left. If
                    omitted, the element occupies the entire cavity.

                sticky: nswe
                    Specifies where the element is placed inside its
                    allocated parcel.

                children: [sublayout... ]
                    Specifies a list of elements to place inside the
                    element. Each element is a tuple (or other sequence)
                    where the first item is the layout name, and the other
                    is a LAYOUT."""
        lspec = None
        if layoutspec:
            lspec = _format_layoutlist(layoutspec)[0]
        elif layoutspec is not None: # will disable the layout ({}, '', etc)
            lspec = "null" # could be any other word, but this may make sense
                           # when calling layout(style) later

        return _list_from_layouttuple(self.tk,
            self.tk.call(self._name, "layout", style, lspec))


    def element_create(self, elementname, etype, *args, **kw):
        """Create a new element in the current theme of given etype."""
        spec, opts = _format_elemcreate(etype, False, *args, **kw)
        self.tk.call(self._name, "element", "create", elementname, etype,
            spec, *opts)


    def element_names(self):
        """Returns the list of elements defined in the current theme."""
        return tuple(n.lstrip('-') for n in self.tk.splitlist(
            self.tk.call(self._name, "element", "names")))


    def element_options(self, elementname):
        """Return the list of elementname's options."""
        return tuple(o.lstrip('-') for o in self.tk.splitlist(
            self.tk.call(self._name, "element", "options", elementname)))


    def theme_create(self, themename, parent=None, settings=None):
        """Creates a new theme.

        It is an error if themename already exists. If parent is
        specified, the new theme will inherit styles, elements and
        layouts from the specified parent theme. If settings are present,
        they are expected to have the same syntax used for theme_settings."""
        script = _script_from_settings(settings) if settings else ''

        if parent:
            self.tk.call(self._name, "theme", "create", themename,
                "-parent", parent, "-settings", script)
        else:
            self.tk.call(self._name, "theme", "create", themename,
                "-settings", script)


    def theme_settings(self, themename, settings):
        """Temporarily sets the current theme to themename, apply specified
        settings and then restore the previous theme.

        Each key in settings is a style and each value may contain the
        keys 'configure', 'map', 'layout' and 'element create' and they
        are expected to have the same format as specified by the methods
        configure, map, layout and element_create respectively."""
        script = _script_from_settings(settings)
        self.tk.call(self._name, "theme", "settings", themename, script)


    def theme_names(self):
        """Returns a list of all known themes."""
        return self.tk.splitlist(self.tk.call(self._name, "theme", "names"))


    def theme_use(self, themename=None):
        """If themename is None, returns the theme in use, otherwise, set
        the current theme to themename, refreshes all widgets and emits
        a <<ThemeChanged>> event."""
        if themename is None:
            # Starting on Tk 8.6, checking this global is no longer needed
            # since it allows doing self.tk.call(self._name, "theme", "use")
            return self.tk.eval("return $ttk::currentTheme")

        # using "ttk::setTheme" instead of "ttk::style theme use" causes
        # the variable currentTheme to be updated, also, ttk::setTheme calls
        # "ttk::style theme use" in order to change theme.
        self.tk.call("ttk::setTheme", themename)


class Widget(tkinter.Widget):
    """Base class for Tk themed widgets."""

    def __init__(self, master, widgetname, kw=None):
        """Constructs a Ttk Widget with the parent master.

        STANDARD OPTIONS

            class, cursor, takefocus, style

        SCROLLABLE WIDGET OPTIONS

            xscrollcommand, yscrollcommand

        LABEL WIDGET OPTIONS

            text, textvariable, underline, image, compound, width

        WIDGET STATES

            active, disabled, focus, pressed, selected, background,
            readonly, alternate, invalid
        """
        master = setup_master(master)
        tkinter.Widget.__init__(self, master, widgetname, kw=kw)


    def identify(self, x, y):
        """Returns the name of the element at position x, y, or the empty
        string if the point does not lie within any element.

        x and y are pixel coordinates relative to the widget."""
        return self.tk.call(self._w, "identify", x, y)


    def instate(self, statespec, callback=None, *args, **kw):
        """Test the widget's state.

        If callback is not specified, returns True if the widget state
        matches statespec and False otherwise. If callback is specified,
        then it will be invoked with *args, **kw if the widget state
        matches statespec. statespec is expected to be a sequence."""
        ret = self.tk.getboolean(
                self.tk.call(self._w, "instate", ' '.join(statespec)))
        if ret and callback is not None:
            return callback(*args, **kw)

        return ret


    def state(self, statespec=None):
        """Modify or inquire widget state.

        Widget state is returned if statespec is None, otherwise it is
        set according to the statespec flags and then a new state spec
        is returned indicating which flags were changed. statespec is
        expected to be a sequence."""
        if statespec is not None:
            statespec = ' '.join(statespec)

        return self.tk.splitlist(str(self.tk.call(self._w, "state", statespec)))


class Button(Widget):
    """Ttk Button widget, displays a textual label and/or image, and
    evaluates a command when pressed."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Button widget with the parent master.

        STANDARD OPTIONS

            class, compound, cursor, image, state, style, takefocus,
            text, textvariable, underline, width

        WIDGET-SPECIFIC OPTIONS

            command, default, width
        """
        Widget.__init__(self, master, "ttk::button", kw)


    def invoke(self):
        """Invokes the command associated with the button."""
        return self.tk.call(self._w, "invoke")


class Checkbutton(Widget):
    """Ttk Checkbutton widget which is either in on- or off-state."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Checkbutton widget with the parent master.

        STANDARD OPTIONS

            class, compound, cursor, image, state, style, takefocus,
            text, textvariable, underline, width

        WIDGET-SPECIFIC OPTIONS

            command, offvalue, onvalue, variable
        """
        Widget.__init__(self, master, "ttk::checkbutton", kw)


    def invoke(self):
        """Toggles between the selected and deselected states and
        invokes the associated command. If the widget is currently
        selected, sets the option variable to the offvalue option
        and deselects the widget; otherwise, sets the option variable
        to the option onvalue.

        Returns the result of the associated command."""
        return self.tk.call(self._w, "invoke")


class Entry(Widget, tkinter.Entry):
    """Ttk Entry widget displays a one-line text string and allows that
    string to be edited by the user."""

    def __init__(self, master=None, widget=None, **kw):
        """Constructs a Ttk Entry widget with the parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus, xscrollcommand

        WIDGET-SPECIFIC OPTIONS

            exportselection, invalidcommand, justify, show, state,
            textvariable, validate, validatecommand, width

        VALIDATION MODES

            none, key, focus, focusin, focusout, all
        """
        Widget.__init__(self, master, widget or "ttk::entry", kw)


    def bbox(self, index):
        """Return a tuple of (x, y, width, height) which describes the
        bounding box of the character given by index."""
        return self._getints(self.tk.call(self._w, "bbox", index))


    def identify(self, x, y):
        """Returns the name of the element at position x, y, or the
        empty string if the coordinates are outside the window."""
        return self.tk.call(self._w, "identify", x, y)


    def validate(self):
        """Force revalidation, independent of the conditions specified
        by the validate option. Returns False if validation fails, True
        if it succeeds. Sets or clears the invalid state accordingly."""
        return self.tk.getboolean(self.tk.call(self._w, "validate"))


class Combobox(Entry):
    """Ttk Combobox widget combines a text field with a pop-down list of
    values."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Combobox widget with the parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            exportselection, justify, height, postcommand, state,
            textvariable, values, width
        """
        Entry.__init__(self, master, "ttk::combobox", **kw)


    def current(self, newindex=None):
        """If newindex is supplied, sets the combobox value to the
        element at position newindex in the list of values. Otherwise,
        returns the index of the current value in the list of values
        or -1 if the current value does not appear in the list."""
        if newindex is None:
            return self.tk.getint(self.tk.call(self._w, "current"))
        return self.tk.call(self._w, "current", newindex)


    def set(self, value):
        """Sets the value of the combobox to value."""
        self.tk.call(self._w, "set", value)


class Frame(Widget):
    """Ttk Frame widget is a container, used to group other widgets
    together."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Frame with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            borderwidth, relief, padding, width, height
        """
        Widget.__init__(self, master, "ttk::frame", kw)


class Label(Widget):
    """Ttk Label widget displays a textual label and/or image."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Label with parent master.

        STANDARD OPTIONS

            class, compound, cursor, image, style, takefocus, text,
            textvariable, underline, width

        WIDGET-SPECIFIC OPTIONS

            anchor, background, font, foreground, justify, padding,
            relief, text, wraplength
        """
        Widget.__init__(self, master, "ttk::label", kw)


class Labelframe(Widget):
    """Ttk Labelframe widget is a container used to group other widgets
    together. It has an optional label, which may be a plain text string
    or another widget."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Labelframe with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS
            labelanchor, text, underline, padding, labelwidget, width,
            height
        """
        Widget.__init__(self, master, "ttk::labelframe", kw)

LabelFrame = Labelframe # tkinter name compatibility


class Menubutton(Widget):
    """Ttk Menubutton widget displays a textual label and/or image, and
    displays a menu when pressed."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Menubutton with parent master.

        STANDARD OPTIONS

            class, compound, cursor, image, state, style, takefocus,
            text, textvariable, underline, width

        WIDGET-SPECIFIC OPTIONS

            direction, menu
        """
        Widget.__init__(self, master, "ttk::menubutton", kw)


class Notebook(Widget):
    """Ttk Notebook widget manages a collection of windows and displays
    a single one at a time. Each child window is associated with a tab,
    which the user may select to change the currently-displayed window."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Notebook with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            height, padding, width

        TAB OPTIONS

            state, sticky, padding, text, image, compound, underline

        TAB IDENTIFIERS (tab_id)

            The tab_id argument found in several methods may take any of
            the following forms:

                * An integer between zero and the number of tabs
                * The name of a child window
                * A positional specification of the form "@x,y", which
                  defines the tab
                * The string "current", which identifies the
                  currently-selected tab
                * The string "end", which returns the number of tabs (only
                  valid for method index)
        """
        Widget.__init__(self, master, "ttk::notebook", kw)


    def add(self, child, **kw):
        """Adds a new tab to the notebook.

        If window is currently managed by the notebook but hidden, it is
        restored to its previous position."""
        self.tk.call(self._w, "add", child, *(_format_optdict(kw)))


    def forget(self, tab_id):
        """Removes the tab specified by tab_id, unmaps and unmanages the
        associated window."""
        self.tk.call(self._w, "forget", tab_id)


    def hide(self, tab_id):
        """Hides the tab specified by tab_id.

        The tab will not be displayed, but the associated window remains
        managed by the notebook and its configuration remembered. Hidden
        tabs may be restored with the add command."""
        self.tk.call(self._w, "hide", tab_id)


    def identify(self, x, y):
        """Returns the name of the tab element at position x, y, or the
        empty string if none."""
        return self.tk.call(self._w, "identify", x, y)


    def index(self, tab_id):
        """Returns the numeric index of the tab specified by tab_id, or
        the total number of tabs if tab_id is the string "end"."""
        return self.tk.getint(self.tk.call(self._w, "index", tab_id))


    def insert(self, pos, child, **kw):
        """Inserts a pane at the specified position.

        pos is either the string end, an integer index, or the name of
        a managed child. If child is already managed by the notebook,
        moves it to the specified position."""
        self.tk.call(self._w, "insert", pos, child, *(_format_optdict(kw)))


    def select(self, tab_id=None):
        """Selects the specified tab.

        The associated child window will be displayed, and the
        previously-selected window (if different) is unmapped. If tab_id
        is omitted, returns the widget name of the currently selected
        pane."""
        return self.tk.call(self._w, "select", tab_id)


    def tab(self, tab_id, option=None, **kw):
        """Query or modify the options of the specific tab_id.

        If kw is not given, returns a dict of the tab option values. If option
        is specified, returns the value of that option. Otherwise, sets the
        options to the corresponding values."""
        if option is not None:
            kw[option] = None
        return _val_or_dict(self.tk, kw, self._w, "tab", tab_id)


    def tabs(self):
        """Returns a list of windows managed by the notebook."""
        return self.tk.splitlist(self.tk.call(self._w, "tabs") or ())


    def enable_traversal(self):
        """Enable keyboard traversal for a toplevel window containing
        this notebook.

        This will extend the bindings for the toplevel window containing
        this notebook as follows:

            Control-Tab: selects the tab following the currently selected
                         one

            Shift-Control-Tab: selects the tab preceding the currently
                               selected one

            Alt-K: where K is the mnemonic (underlined) character of any
                   tab, will select that tab.

        Multiple notebooks in a single toplevel may be enabled for
        traversal, including nested notebooks. However, notebook traversal
        only works properly if all panes are direct children of the
        notebook."""
        # The only, and good, difference I see is about mnemonics, which works
        # after calling this method. Control-Tab and Shift-Control-Tab always
        # works (here at least).
        self.tk.call("ttk::notebook::enableTraversal", self._w)


class Panedwindow(Widget, tkinter.PanedWindow):
    """Ttk Panedwindow widget displays a number of subwindows, stacked
    either vertically or horizontally."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Panedwindow with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            orient, width, height

        PANE OPTIONS

            weight
        """
        Widget.__init__(self, master, "ttk::panedwindow", kw)


    forget = tkinter.PanedWindow.forget # overrides Pack.forget


    def insert(self, pos, child, **kw):
        """Inserts a pane at the specified positions.

        pos is either the string end, and integer index, or the name
        of a child. If child is already managed by the paned window,
        moves it to the specified position."""
        self.tk.call(self._w, "insert", pos, child, *(_format_optdict(kw)))


    def pane(self, pane, option=None, **kw):
        """Query or modify the options of the specified pane.

        pane is either an integer index or the name of a managed subwindow.
        If kw is not given, returns a dict of the pane option values. If
        option is specified then the value for that option is returned.
        Otherwise, sets the options to the corresponding values."""
        if option is not None:
            kw[option] = None
        return _val_or_dict(self.tk, kw, self._w, "pane", pane)


    def sashpos(self, index, newpos=None):
        """If newpos is specified, sets the position of sash number index.

        May adjust the positions of adjacent sashes to ensure that
        positions are monotonically increasing. Sash positions are further
        constrained to be between 0 and the total size of the widget.

        Returns the new position of sash number index."""
        return self.tk.getint(self.tk.call(self._w, "sashpos", index, newpos))

PanedWindow = Panedwindow # tkinter name compatibility


class Progressbar(Widget):
    """Ttk Progressbar widget shows the status of a long-running
    operation. They can operate in two modes: determinate mode shows the
    amount completed relative to the total amount of work to be done, and
    indeterminate mode provides an animated display to let the user know
    that something is happening."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Progressbar with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            orient, length, mode, maximum, value, variable, phase
        """
        Widget.__init__(self, master, "ttk::progressbar", kw)


    def start(self, interval=None):
        """Begin autoincrement mode: schedules a recurring timer event
        that calls method step every interval milliseconds.

        interval defaults to 50 milliseconds (20 steps/second) if omitted."""
        self.tk.call(self._w, "start", interval)


    def step(self, amount=None):
        """Increments the value option by amount.

        amount defaults to 1.0 if omitted."""
        self.tk.call(self._w, "step", amount)


    def stop(self):
        """Stop autoincrement mode: cancels any recurring timer event
        initiated by start."""
        self.tk.call(self._w, "stop")


class Radiobutton(Widget):
    """Ttk Radiobutton widgets are used in groups to show or change a
    set of mutually-exclusive options."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Radiobutton with parent master.

        STANDARD OPTIONS

            class, compound, cursor, image, state, style, takefocus,
            text, textvariable, underline, width

        WIDGET-SPECIFIC OPTIONS

            command, value, variable
        """
        Widget.__init__(self, master, "ttk::radiobutton", kw)


    def invoke(self):
        """Sets the option variable to the option value, selects the
        widget, and invokes the associated command.

        Returns the result of the command, or an empty string if
        no command is specified."""
        return self.tk.call(self._w, "invoke")


class Scale(Widget, tkinter.Scale):
    """Ttk Scale widget is typically used to control the numeric value of
    a linked variable that varies uniformly over some range."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Scale with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            command, from, length, orient, to, value, variable
        """
        Widget.__init__(self, master, "ttk::scale", kw)


    def configure(self, cnf=None, **kw):
        """Modify or query scale options.

        Setting a value for any of the "from", "from_" or "to" options
        generates a <<RangeChanged>> event."""
        retval = Widget.configure(self, cnf, **kw)
        if not isinstance(cnf, (type(None), str)):
            kw.update(cnf)
        if any(['from' in kw, 'from_' in kw, 'to' in kw]):
            self.event_generate('<<RangeChanged>>')
        return retval


    def get(self, x=None, y=None):
        """Get the current value of the value option, or the value
        corresponding to the coordinates x, y if they are specified.

        x and y are pixel coordinates relative to the scale widget
        origin."""
        return self.tk.call(self._w, 'get', x, y)


class Scrollbar(Widget, tkinter.Scrollbar):
    """Ttk Scrollbar controls the viewport of a scrollable widget."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Scrollbar with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            command, orient
        """
        Widget.__init__(self, master, "ttk::scrollbar", kw)


class Separator(Widget):
    """Ttk Separator widget displays a horizontal or vertical separator
    bar."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Separator with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus

        WIDGET-SPECIFIC OPTIONS

            orient
        """
        Widget.__init__(self, master, "ttk::separator", kw)


class Sizegrip(Widget):
    """Ttk Sizegrip allows the user to resize the containing toplevel
    window by pressing and dragging the grip."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Sizegrip with parent master.

        STANDARD OPTIONS

            class, cursor, state, style, takefocus
        """
        Widget.__init__(self, master, "ttk::sizegrip", kw)


class Spinbox(Entry):
    """Ttk Spinbox is an Entry with increment and decrement arrows

    It is commonly used for number entry or to select from a list of
    string values.
    """

    def __init__(self, master=None, **kw):
        """Construct a Ttk Spinbox widget with the parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus, validate,
            validatecommand, xscrollcommand, invalidcommand

        WIDGET-SPECIFIC OPTIONS

            to, from_, increment, values, wrap, format, command
        """
        Entry.__init__(self, master, "ttk::spinbox", **kw)


    def set(self, value):
        """Sets the value of the Spinbox to value."""
        self.tk.call(self._w, "set", value)


class Treeview(Widget, tkinter.XView, tkinter.YView):
    """Ttk Treeview widget displays a hierarchical collection of items.

    Each item has a textual label, an optional image, and an optional list
    of data values. The data values are displayed in successive columns
    after the tree label."""

    def __init__(self, master=None, **kw):
        """Construct a Ttk Treeview with parent master.

        STANDARD OPTIONS

            class, cursor, style, takefocus, xscrollcommand,
            yscrollcommand

        WIDGET-SPECIFIC OPTIONS

            columns, displaycolumns, height, padding, selectmode, show

        ITEM OPTIONS

            text, image, values, open, tags

        TAG OPTIONS

            foreground, background, font, image
        """
        Widget.__init__(self, master, "ttk::treeview", kw)


    def bbox(self, item, column=None):
        """Returns the bounding box (relative to the treeview widget's
        window) of the specified item in the form x y width height.

        If column is specified, returns the bounding box of that cell.
        If the item is not visible (i.e., if it is a descendant of a
        closed item or is scrolled offscreen), returns an empty string."""
        return self._getints(self.tk.call(self._w, "bbox", item, column)) or ''


    def get_children(self, item=None):
        """Returns a tuple of children belonging to item.

        If item is not specified, returns root children."""
        return self.tk.splitlist(
                self.tk.call(self._w, "children", item or '') or ())


    def set_children(self, item, *newchildren):
        """Replaces item's child with newchildren.

        Children present in item that are not present in newchildren
        are detached from tree. No items in newchildren may be an
        ancestor of item."""
        self.tk.call(self._w, "children", item, newchildren)


    def column(self, column, option=None, **kw):
        """Query or modify the options for the specified column.

        If kw is not given, returns a dict of the column option values. If
        option is specified then the value for that option is returned.
        Otherwise, sets the options to the corresponding values."""
        if option is not None:
            kw[option] = None
        return _val_or_dict(self.tk, kw, self._w, "column", column)


    def delete(self, *items):
        """Delete all specified items and all their descendants. The root
        item may not be deleted."""
        self.tk.call(self._w, "delete", items)


    def detach(self, *items):
        """Unlinks all of the specified items from the tree.

        The items and all of their descendants are still present, and may
        be reinserted at another point in the tree, but will not be
        displayed. The root item may not be detached."""
        self.tk.call(self._w, "detach", items)


    def exists(self, item):
        """Returns True if the specified item is present in the tree,
        False otherwise."""
        return self.tk.getboolean(self.tk.call(self._w, "exists", item))


    def focus(self, item=None):
        """If item is specified, sets the focus item to item. Otherwise,
        returns the current focus item, or '' if there is none."""
        return self.tk.call(self._w, "focus", item)


    def heading(self, column, option=None, **kw):
        """Query or modify the heading options for the specified column.

        If kw is not given, returns a dict of the heading option values. If
        option is specified then the value for that option is returned.
        Otherwise, sets the options to the corresponding values.

        Valid options/values are:
            text: text
                The text to display in the column heading
            image: image_name
                Specifies an image to display to the right of the column
                heading
            anchor: anchor
                Specifies how the heading text should be aligned. One of
                the standard Tk anchor values
            command: callback
                A callback to be invoked when the heading label is
                pressed.

        To configure the tree column heading, call this with column = "#0" """
        cmd = kw.get('command')
        if cmd and not isinstance(cmd, str):
            # callback not registered yet, do it now
            kw['command'] = self.master.register(cmd, self._substitute)

        if option is not None:
            kw[option] = None

        return _val_or_dict(self.tk, kw, self._w, 'heading', column)


    def identify(self, component, x, y):
        """Returns a description of the specified component under the
        point given by x and y, or the empty string if no such component
        is present at that position."""
        return self.tk.call(self._w, "identify", component, x, y)


    def identify_row(self, y):
        """Returns the item ID of the item at position y."""
        return self.identify("row", 0, y)


    def identify_column(self, x):
        """Returns the data column identifier of the cell at position x.

        The tree column has ID #0."""
        return self.identify("column", x, 0)


    def identify_region(self, x, y):
        """Returns one of:

        heading: Tree heading area.
        separator: Space between two columns headings;
        tree: The tree area.
        cell: A data cell.

        * Availability: Tk 8.6"""
        return self.identify("region", x, y)


    def identify_element(self, x, y):
        """Returns the element at position x, y.

        * Availability: Tk 8.6"""
        return self.identify("element", x, y)


    def index(self, item):
        """Returns the integer index of item within its parent's list
        of children."""
        return self.tk.getint(self.tk.call(self._w, "index", item))


    def insert(self, parent, index, iid=None, **kw):
        """Creates a new item and return the item identifier of the newly
        created item.

        parent is the item ID of the parent item, or the empty string
        to create a new top-level item. index is an integer, or the value
        end, specifying where in the list of parent's children to insert
        the new item. If index is less than or equal to zero, the new node
        is inserted at the beginning, if index is greater than or equal to
        the current number of children, it is inserted at the end. If iid
        is specified, it is used as the item identifier, iid must not
        already exist in the tree. Otherwise, a new unique identifier
        is generated."""
        opts = _format_optdict(kw)
        if iid is not None:
            res = self.tk.call(self._w, "insert", parent, index,
                "-id", iid, *opts)
        else:
            res = self.tk.call(self._w, "insert", parent, index, *opts)

        return res


    def item(self, item, option=None, **kw):
        """Query or modify the options for the specified item.

        If no options are given, a dict with options/values for the item
        is returned. If option is specified then the value for that option
        is returned. Otherwise, sets the options to the corresponding
        values as given by kw."""
        if option is not None:
            kw[option] = None
        return _val_or_dict(self.tk, kw, self._w, "item", item)


    def move(self, item, parent, index):
        """Moves item to position index in parent's list of children.

        It is illegal to move an item under one of its descendants. If
        index is less than or equal to zero, item is moved to the
        beginning, if greater than or equal to the number of children,
        it is moved to the end. If item was detached it is reattached."""
        self.tk.call(self._w, "move", item, parent, index)

    reattach = move # A sensible method name for reattaching detached items


    def next(self, item):
        """Returns the identifier of item's next sibling, or '' if item
        is the last child of its parent."""
        return self.tk.call(self._w, "next", item)


    def parent(self, item):
        """Returns the ID of the parent of item, or '' if item is at the
        top level of the hierarchy."""
        return self.tk.call(self._w, "parent", item)


    def prev(self, item):
        """Returns the identifier of item's previous sibling, or '' if
        item is the first child of its parent."""
        return self.tk.call(self._w, "prev", item)


    def see(self, item):
        """Ensure that item is visible.

        Sets all of item's ancestors open option to True, and scrolls
        the widget if necessary so that item is within the visible
        portion of the tree."""
        self.tk.call(self._w, "see", item)


    def selection(self):
        """Returns the tuple of selected items."""
        return self.tk.splitlist(self.tk.call(self._w, "selection"))


    def _selection(self, selop, items):
        if len(items) == 1 and isinstance(items[0], (tuple, list)):
            items = items[0]

        self.tk.call(self._w, "selection", selop, items)


    def selection_set(self, *items):
        """The specified items becomes the new selection."""
        self._selection("set", items)


    def selection_add(self, *items):
        """Add all of the specified items to the selection."""
        self._selection("add", items)


    def selection_remove(self, *items):
        """Remove all of the specified items from the selection."""
        self._selection("remove", items)


    def selection_toggle(self, *items):
        """Toggle the selection state of each specified item."""
        self._selection("toggle", items)


    def set(self, item, column=None, value=None):
        """Query or set the value of given item.

        With one argument, return a dictionary of column/value pairs
        for the specified item. With two arguments, return the current
        value of the specified column. With three arguments, set the
        value of given column in given item to the specified value."""
        res = self.tk.call(self._w, "set", item, column, value)
        if column is None and value is None:
            return _splitdict(self.tk, res,
                              cut_minus=False, conv=_tclobj_to_py)
        else:
            return res


    def tag_bind(self, tagname, sequence=None, callback=None):
        """Bind a callback for the given event sequence to the tag tagname.
        When an event is delivered to an item, the callbacks for each
        of the item's tags option are called."""
        self._bind((self._w, "tag", "bind", tagname), sequence, callback, add=0)


    def tag_configure(self, tagname, option=None, **kw):
        """Query or modify the options for the specified tagname.

        If kw is not given, returns a dict of the option settings for tagname.
        If option is specified, returns the value for that option for the
        specified tagname. Otherwise, sets the options to the corresponding
        values for the given tagname."""
        if option is not None:
            kw[option] = None
        return _val_or_dict(self.tk, kw, self._w, "tag", "configure",
            tagname)


    def tag_has(self, tagname, item=None):
        """If item is specified, returns 1 or 0 depending on whether the
        specified item has the given tagname. Otherwise, returns a list of
        all items which have the specified tag.

        * Availability: Tk 8.6"""
        if item is None:
            return self.tk.splitlist(
                self.tk.call(self._w, "tag", "has", tagname))
        else:
            return self.tk.getboolean(
                self.tk.call(self._w, "tag", "has", tagname, item))


# Extensions

class LabeledScale(Frame):
    """A Ttk Scale widget with a Ttk Label widget indicating its
    current value.

    The Ttk Scale can be accessed through instance.scale, and Ttk Label
    can be accessed through instance.label"""

    def __init__(self, master=None, variable=None, from_=0, to=10, **kw):
        """Construct a horizontal LabeledScale with parent master, a
        variable to be associated with the Ttk Scale widget and its range.
        If variable is not specified, a tkinter.IntVar is created.

        WIDGET-SPECIFIC OPTIONS

            compound: 'top' or 'bottom'
                Specifies how to display the label relative to the scale.
                Defaults to 'top'.
        """
        self._label_top = kw.pop('compound', 'top') == 'top'

        Frame.__init__(self, master, **kw)
        self._variable = variable or tkinter.IntVar(master)
        self._variable.set(from_)
        self._last_valid = from_

        self.label = Label(self)
        self.scale = Scale(self, variable=self._variable, from_=from_, to=to)
        self.scale.bind('<<RangeChanged>>', self._adjust)

        # position scale and label according to the compound option
        scale_side = 'bottom' if self._label_top else 'top'
        label_side = 'top' if scale_side == 'bottom' else 'bottom'
        self.scale.pack(side=scale_side, fill='x')
        # Dummy required to make frame correct height
        dummy = Label(self)
        dummy.pack(side=label_side)
        dummy.lower()
        self.label.place(anchor='n' if label_side == 'top' else 's')

        # update the label as scale or variable changes
        self.__tracecb = self._variable.trace_variable('w', self._adjust)
        self.bind('<Configure>', self._adjust)
        self.bind('<Map>', self._adjust)


    def destroy(self):
        """Destroy this widget and possibly its associated variable."""
        try:
            self._variable.trace_vdelete('w', self.__tracecb)
        except AttributeError:
            pass
        else:
            del self._variable
        super().destroy()
        self.label = None
        self.scale = None


    def _adjust(self, *args):
        """Adjust the label position according to the scale."""
        def adjust_label():
            self.update_idletasks() # "force" scale redraw

            x, y = self.scale.coords()
            if self._label_top:
                y = self.scale.winfo_y() - self.label.winfo_reqheight()
            else:
                y = self.scale.winfo_reqheight() + self.label.winfo_reqheight()

            self.label.place_configure(x=x, y=y)

        from_ = _to_number(self.scale['from'])
        to = _to_number(self.scale['to'])
        if to < from_:
            from_, to = to, from_
        newval = self._variable.get()
        if not from_ <= newval <= to:
            # value outside range, set value back to the last valid one
            self.value = self._last_valid
            return

        self._last_valid = newval
        self.label['text'] = newval
        self.after_idle(adjust_label)

    @property
    def value(self):
        """Return current scale value."""
        return self._variable.get()

    @value.setter
    def value(self, val):
        """Set new scale value."""
        self._variable.set(val)


class OptionMenu(Menubutton):
    """Themed OptionMenu, based after tkinter's OptionMenu, which allows
    the user to select a value from a menu."""

    def __init__(self, master, variable, default=None, *values, **kwargs):
        """Construct a themed OptionMenu widget with master as the parent,
        the resource textvariable set to variable, the initially selected
        value specified by the default parameter, the menu values given by
        *values and additional keywords.

        WIDGET-SPECIFIC OPTIONS

            style: stylename
                Menubutton style.
            direction: 'above', 'below', 'left', 'right', or 'flush'
                Menubutton direction.
            command: callback
                A callback that will be invoked after selecting an item.
        """
        kw = {'textvariable': variable, 'style': kwargs.pop('style', None),
              'direction': kwargs.pop('direction', None)}
        Menubutton.__init__(self, master, **kw)
        self['menu'] = tkinter.Menu(self, tearoff=False)

        self._variable = variable
        self._callback = kwargs.pop('command', None)
        if kwargs:
            raise tkinter.TclError('unknown option -%s' % (
                next(iter(kwargs.keys()))))

        self.set_menu(default, *values)


    def __getitem__(self, item):
        if item == 'menu':
            return self.nametowidget(Menubutton.__getitem__(self, item))

        return Menubutton.__getitem__(self, item)


    def set_menu(self, default=None, *values):
        """Build a new menu of radiobuttons with *values and optionally
        a default value."""
        menu = self['menu']
        menu.delete(0, 'end')
        for val in values:
            menu.add_radiobutton(label=val,
                command=(
                    None if self._callback is None
                    else lambda val=val: self._callback(val)
                ),
                variable=self._variable)

        if default:
            self._variable.set(default)


    def destroy(self):
        """Destroy this widget and its associated variable."""
        try:
            del self._variable
        except AttributeError:
            pass
        super().destroy()
