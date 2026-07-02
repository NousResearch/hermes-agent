"""
A number of functions that enhance IDLE on macOS.
"""
from os.path import expanduser
import plistlib
from sys import platform  # Used in _init_tk_type, changed by test.

import tkinter


## Define functions that query the Mac graphics type.
## _tk_type and its initializer are private to this section.

_tk_type = None

def _init_tk_type():
    """ Initialize _tk_type for isXyzTk functions.

    This function is only called once, when _tk_type is still None.
    """
    global _tk_type
    if platform == 'darwin':

        # When running IDLE, GUI is present, test/* may not be.
        # When running tests, test/* is present, GUI may not be.
        # If not, guess most common.  Does not matter for testing.
        from idlelib.__init__ import testing
        if testing:
            from test.support import requires, ResourceDenied
            try:
                requires('gui')
            except ResourceDenied:
                _tk_type = "cocoa"
                return

        root = tkinter.Tk()
        ws = root.tk.call('tk', 'windowingsystem')
        if 'x11' in ws:
            _tk_type = "xquartz"
        elif 'aqua' not in ws:
            _tk_type = "other"
        elif 'AppKit' in root.tk.call('winfo', 'server', '.'):
            _tk_type = "cocoa"
        else:
            _tk_type = "carbon"
        root.destroy()
    else:
        _tk_type = "other"
    return

def isAquaTk():
    """
    Returns True if IDLE is using a native OS X Tk (Cocoa or Carbon).
    """
    if not _tk_type:
        _init_tk_type()
    return _tk_type == "cocoa" or _tk_type == "carbon"

def isCarbonTk():
    """
    Returns True if IDLE is using a Carbon Aqua Tk (instead of the
    newer Cocoa Aqua Tk).
    """
    if not _tk_type:
        _init_tk_type()
    return _tk_type == "carbon"

def isCocoaTk():
    """
    Returns True if IDLE is using a Cocoa Aqua Tk.
    """
    if not _tk_type:
        _init_tk_type()
    return _tk_type == "cocoa"

def isXQuartz():
    """
    Returns True if IDLE is using an OS X X11 Tk.
    """
    if not _tk_type:
        _init_tk_type()
    return _tk_type == "xquartz"


def readSystemPreferences():
    """
    Fetch the macOS system preferences.
    """
    if platform != 'darwin':
        return None

    plist_path = expanduser('~/Library/Preferences/.GlobalPreferences.plist')
    try:
        with open(plist_path, 'rb') as plist_file:
            return plistlib.load(plist_file)
    except OSError:
        return None


def preferTabsPreferenceWarning():
    """
    Warn if "Prefer tabs when opening documents" is set to "Always".
    """
    if platform != 'darwin':
        return None

    prefs = readSystemPreferences()
    if prefs and prefs.get('AppleWindowTabbingMode') == 'always':
        return (
            'WARNING: The system preference "Prefer tabs when opening'
            ' documents" is set to "Always". This will cause various problems'
            ' with IDLE. For the best experience, change this setting when'
            ' running IDLE (via System Preferences -> Dock).'
        )
    return None


## Fix the menu and related functions.

def addOpenEventSupport(root, flist):
    """
    This ensures that the application will respond to open AppleEvents, which
    makes is feasible to use IDLE as the default application for python files.
    """
    def doOpenFile(*args):
        for fn in args:
            flist.open(fn)

    # The command below is a hook in aquatk that is called whenever the app
    # receives a file open event. The callback can have multiple arguments,
    # one for every file that should be opened.
    root.createcommand("::tk::mac::OpenDocument", doOpenFile)

def hideTkConsole(root):
    try:
        root.tk.call('console', 'hide')
    except tkinter.TclError:
        # Some versions of the Tk framework don't have a console object
        pass

def overrideRootMenu(root, flist):
    """
    Replace the Tk root menu by something that is more appropriate for
    IDLE with an Aqua Tk.
    """
    # The menu that is attached to the Tk root (".") is also used by AquaTk for
    # all windows that don't specify a menu of their own. The default menubar
    # contains a number of menus, none of which are appropriate for IDLE. The
    # Most annoying of those is an 'About Tck/Tk...' menu in the application
    # menu.
    #
    # This function replaces the default menubar by a mostly empty one, it
    # should only contain the correct application menu and the window menu.
    #
    # Due to a (mis-)feature of TkAqua the user will also see an empty Help
    # menu.
    from tkinter import Menu
    from idlelib import mainmenu
    from idlelib import window

    closeItem = mainmenu.menudefs[0][1][-2]

    # Remove the last 3 items of the file menu: a separator, close window and
    # quit. Close window will be reinserted just above the save item, where
    # it should be according to the HIG. Quit is in the application menu.
    del mainmenu.menudefs[0][1][-3:]
    mainmenu.menudefs[0][1].insert(6, closeItem)

    # Remove the 'About' entry from the help menu, it is in the application
    # menu
    del mainmenu.menudefs[-1][1][0:2]
    # Remove the 'Configure Idle' entry from the options menu, it is in the
    # application menu as 'Preferences'
    del mainmenu.menudefs[-3][1][0:2]
    menubar = Menu(root)
    root.configure(menu=menubar)
    menudict = {}

    menudict['window'] = menu = Menu(menubar, name='window', tearoff=0)
    menubar.add_cascade(label='Window', menu=menu, underline=0)

    def postwindowsmenu(menu=menu):
        end = menu.index('end')
        if end is None:
            end = -1

        if end > 0:
            menu.delete(0, end)
        window.add_windows_to_menu(menu)
    window.register_callback(postwindowsmenu)

    def about_dialog(event=None):
        "Handle Help 'About IDLE' event."
        # Synchronize with editor.EditorWindow.about_dialog.
        from idlelib import help_about
        help_about.AboutDialog(root)

    def config_dialog(event=None):
        "Handle Options 'Configure IDLE' event."
        # Synchronize with editor.EditorWindow.config_dialog.
        from idlelib import configdialog

        # Ensure that the root object has an instance_dict attribute,
        # mirrors code in EditorWindow (although that sets the attribute
        # on an EditorWindow instance that is then passed as the first
        # argument to ConfigDialog)
        root.instance_dict = flist.inversedict
        configdialog.ConfigDialog(root, 'Settings')

    def help_dialog(event=None):
        "Handle Help 'IDLE Help' event."
        # Synchronize with editor.EditorWindow.help_dialog.
        from idlelib import help
        help.show_idlehelp(root)

    root.bind('<<about-idle>>', about_dialog)
    root.bind('<<open-config-dialog>>', config_dialog)
    root.createcommand('::tk::mac::ShowPreferences', config_dialog)
    if flist:
        root.bind('<<close-all-windows>>', flist.close_all_callback)

        # The binding above doesn't reliably work on all versions of Tk
        # on macOS. Adding command definition below does seem to do the
        # right thing for now.
        root.createcommand('::tk::mac::Quit', flist.close_all_callback)

    if isCarbonTk():
        # for Carbon AquaTk, replace the default Tk apple menu
        menudict['application'] = menu = Menu(menubar, name='apple',
                                              tearoff=0)
        menubar.add_cascade(label='IDLE', menu=menu)
        mainmenu.menudefs.insert(0,
            ('application', [
                ('About IDLE', '<<about-idle>>'),
                    None,
                ]))
    if isCocoaTk():
        # replace default About dialog with About IDLE one
        root.createcommand('tkAboutDialog', about_dialog)
        # replace default "Help" item in Help menu
        root.createcommand('::tk::mac::ShowHelp', help_dialog)
        # remove redundant "IDLE Help" from menu
        del mainmenu.menudefs[-1][1][0]

def fixb2context(root):
    '''Removed bad AquaTk Button-2 (right) and Paste bindings.

    They prevent context menu access and seem to be gone in AquaTk8.6.
    See issue #24801.
    '''
    root.unbind_class('Text', '<B2>')
    root.unbind_class('Text', '<B2-Motion>')
    root.unbind_class('Text', '<<PasteSelection>>')

def setupApp(root, flist):
    """
    Perform initial OS X customizations if needed.
    Called from pyshell.main() after initial calls to Tk()

    There are currently three major versions of Tk in use on OS X:
        1. Aqua Cocoa Tk (native default since OS X 10.6)
        2. Aqua Carbon Tk (original native, 32-bit only, deprecated)
        3. X11 (supported by some third-party distributors, deprecated)
    There are various differences among the three that affect IDLE
    behavior, primarily with menus, mouse key events, and accelerators.
    Some one-time customizations are performed here.
    Others are dynamically tested throughout idlelib by calls to the
    isAquaTk(), isCarbonTk(), isCocoaTk(), isXQuartz() functions which
    are initialized here as well.
    """
    if isAquaTk():
        hideTkConsole(root)
        overrideRootMenu(root, flist)
        addOpenEventSupport(root, flist)
        fixb2context(root)


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_macosx', verbosity=2)
