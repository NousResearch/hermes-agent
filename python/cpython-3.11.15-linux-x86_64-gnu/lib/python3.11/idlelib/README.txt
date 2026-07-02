README.txt: an index to idlelib files and the IDLE menu.

IDLE is Python's Integrated Development and Learning
Environment.  The user documentation is part of the Library Reference and
is available in IDLE by selecting Help => IDLE Help.  This README documents
idlelib for IDLE developers and curious users.

IDLELIB FILES lists files alphabetically by category,
with a short description of each.

IDLE MENU show the menu tree, annotated with the module
or module object that implements the corresponding function.

This file is descriptive, not prescriptive, and may have errors
and omissions and lag behind changes in idlelib.


IDLELIB FILES
=============

Implementation files not in IDLE MENU are marked (nim).

Startup
-------
__init__.py  # import, does nothing
__main__.py  # -m, starts IDLE
idle.bat
idle.py
idle.pyw

Implementation
--------------
autocomplete.py   # Complete attribute names or filenames.
autocomplete_w.py # Display completions.
autoexpand.py     # Expand word with previous word in file.
browser.py        # Create module browser window.
calltip.py        # Create calltip text.
calltip_w.py      # Display calltip.
codecontext.py    # Show compound statement headers otherwise not visible.
colorizer.py      # Colorize text (nim).
config.py         # Load, fetch, and save configuration (nim).
configdialog.py   # Display user configuration dialogs.
config_key.py     # Change keybindings.
debugger.py       # Debug code run from shell or editor; show window.
debugger_r.py     # Debug code run in remote process.
debugobj.py       # Define class used in stackviewer.
debugobj_r.py     # Communicate objects between processes with rpc (nim).
delegator.py      # Define base class for delegators (nim).
dynoption.py      # Define mutable OptionMenu widget (nim)
editor.py         # Define most of editor and utility functions.
filelist.py       # Open files and manage list of open windows (nim).
format.py         # Define format menu options.
grep.py           # Find all occurrences of pattern in multiple files.
help.py           # Display IDLE's html doc.
help_about.py     # Display About IDLE dialog.
history.py        # Get previous or next user input in shell (nim)
hyperparser.py    # Parse code around a given index.
iomenu.py         # Open, read, and write files
macosx.py         # Help IDLE run on Macs (nim).
mainmenu.py       # Define most of IDLE menu.
multicall.py      # Wrap tk widget to allow multiple calls per event (nim).
outwin.py         # Create window for grep output.
parenmatch.py     # Match fenceposts: (), [], and {}.
pathbrowser.py    # Create path browser window.
percolator.py     # Manage delegator stack (nim).
pyparse.py        # Give information on code indentation
pyshell.py        # Start IDLE, manage shell, complete editor window
query.py          # Query user for information
redirector.py     # Intercept widget subcommands (for percolator) (nim).
replace.py        # Search and replace pattern in text.
rpc.py            # Communicate between idle and user processes (nim).
run.py            # Manage user code execution subprocess.
runscript.py      # Check and run user code.
scrolledlist.py   # Define scrolledlist widget for IDLE (nim).
search.py         # Search for pattern in text.
searchbase.py     # Define base for search, replace, and grep dialogs.
searchengine.py   # Define engine for all 3 search dialogs.
sidebar.py        # Define line number and shell prompt sidebars.
squeezer.py       # Squeeze long shell output (nim).
stackviewer.py    # View stack after exception.
statusbar.py      # Define status bar for windows (nim).
tabbedpages.py    # Define tabbed pages widget (nim).
textview.py       # Define read-only text widget (nim).
tooltip.py        # Define popups for calltips, squeezer (nim).
tree.py           # Define tree widget, used in browsers (nim).
undo.py           # Manage undo stack.
util.py           # Define common objects imported elsewhere (nim).
windows.py        # Manage window list and define listed top level.
zoomheight.py     # Zoom window to full height of screen.
zzdummy.py        # Example extension.

Configuration
-------------
config-extensions.def # Defaults for extensions
config-highlight.def  # Defaults for colorizing
config-keys.def       # Defaults for key bindings
config-main.def       # Defaults for font and general tabs

Text
----
CREDITS.txt  # not maintained, displayed by About IDLE
HISTORY.txt  # NEWS up to July 2001
NEWS.txt     # commits, displayed by About IDLE
NEWS2.txt    # commits to Python2
README.txt   # this file, displayed by About IDLE
TODO.txt     # needs review
extend.txt   # about writing extensions
help.html    # copy of idle.html in docs, displayed by IDLE Help

Subdirectories
--------------
Icons        # small image files
idle_test    # files for human test and automated unit tests


IDLE MENUS
==========

Top level items and most submenu items are defined in mainmenu.
Extensions add submenu items when active.  The names given are
found, quoted, in one of these modules, paired with a '<<pseudoevent>>'.
Each pseudoevent is bound to an event handler.  Some event handlers
call another function that does the actual work.  The annotations below
are intended to at least give the module where the actual work is done.
'eEW' = editor.EditorWindow

File
  New File         # eEW.new_callback
  Open...          # iomenu.open
  Open Module      # eEw.open_module
  Recent Files
  Class Browser    # eEW.open_class_browser, browser.ClassBrowser
  Path Browser     # eEW.open_path_browser, pathbrowser
  ---
  Save             # iomenu.save
  Save As...       # iomenu.save_as
  Save Copy As...  # iomenu.save_a_copy
  ---
  Print Window     # iomenu.print_window
  ---
  Close            # eEW.close_event
  Exit             # flist.close_all_callback (bound in eEW)

Edit
  Undo             # undodelegator
  Redo             # undodelegator
  ---              # eEW.right_menu_event
  Cut              # eEW.cut
  Copy             # eEW.copy
  Paste            # eEW.past
  Select All       # eEW.select_all (+ see eEW.remove_selection)
  ---              # Next 5 items use searchengine; dialogs use searchbase
  Find             # eEW.find_event, search.SearchDialog.find
  Find Again       # eEW.find_again_event, sSD.find_again
  Find Selection   # eEW.find_selection_event, sSD.find_selection
  Find in Files... # eEW.find_in_files_event, grep
  Replace...       # eEW.replace_event, replace.ReplaceDialog.replace
  Go to Line       # eEW.goto_line_event
  Show Completions # autocomplete extension and autocompleteWidow (&HP)
  Expand Word      # autoexpand extension
  Show call tip    # Calltips extension and CalltipWindow (& Hyperparser)
  Show surrounding parens  # parenmatch (& Hyperparser)

Format (Editor only) [fFR = format.FormatRegion]
  Format Paragraph # format.FormatParagraph.format_paragraph_event
  Indent Region    # fFR.indent_region_event
  Dedent Region    # fFR.dedent_region_event
  Comment Out Reg. # fFR.comment_region_event
  Uncomment Region # fFR.uncomment_region_event
  Tabify Region    # fFR.tabify_region_event
  Untabify Region  # fFR.untabify_region_event
  Toggle Tabs      # format.Indents.toggle_tabs_event
  New Indent Width # format.Indents.change_indentwidth_event
  Strip tailing whitespace  # format.rstrip
  Zin              # zzdummy
  Zout             # zzdummy

Run (Editor only)
  Run Module         # runscript.ScriptBinding.run_module_event
  Run... Customized  # runscript.ScriptBinding.run_custom_event
  Check Module       # runscript.ScriptBinding.check_module_event
  Python Shell       # pyshell.Pyshell, pyshell.ModifiedInterpreter

Shell  # pyshell
  View Last Restart    # pyshell.PyShell.view_restart_mark
  Restart Shell        # pyshell.PyShell.restart_shell
  Previous History     # history.History.history_prev
  Next History         # history.History.history_next
  Interrupt Execution  # pyshell.PyShell.cancel_callback

Debug (Shell only)
  Go to File/Line  # outwin.OutputWindow.goto_file_line
  debugger         # debugger, debugger_r, PyShell.toggle_debugger
  Stack Viewer     # stackviewer, PyShell.open_stack_viewer
  Auto-open Stack Viewer  # stackviewer

Options
  Configure IDLE   # eEW.config_dialog, config, configdialog (cd)
    (Parts of the dialog)
    Buttons        # cd.ConfigDialog
    Font tab       # cd.FontPage, config-main.def
    Highlight tab  # cd.HighPage, query, config-highlight.def
    Keys tab       # cd.KeysPage, query, config_key, config_keys.def
    Windows tab    # cd.WinPage, config_main.def
    Shell/Ed tab   # cd.ShedPage, config-main.def
    Extensions tab # config-extensions.def, corresponding .py files
  ---
  ... Code Context # codecontext
  ... Line Numbers # sidebar
  Zoomheight       # zoomheight

Window
  <open windows>   # windows

Help
  About IDLE       # eEW.about_dialog, help_about.AboutDialog
  ---
  IDLE Help        # eEW.help_dialog, help.show_idlehelp
  Python Docs      # eEW.python_docs
  Turtle Demo      # eEW.open_turtle_demo
  ---
  <other help sources>

<Context Menu> (right click)
  Defined in editor, PyShell.pyshell
    Cut
    Copy
    Paste
    ---
    Go to file/line (shell and output only)
    Set Breakpoint (editor only)
    Clear Breakpoint (editor only)
  Defined in debugger
    Go to source line
    Show stack frame

<No menu>
Center Insert      # eEW.center_insert_event


OTHER TOPICS
============

Generally use PEP 8.

import statements
-----------------
Put imports at the top, unless there is a good reason otherwise.
PEP 8 says to group stdlib, 3rd-party dependencies, and package imports.
For idlelib, the groups are general stdlib, tkinter, and idlelib.
Sort modules within each group, except that tkinter.ttk follows tkinter.
Sort 'from idlelib import mod1' and 'from idlelib.mod2 import object'
together by module, ignoring within module objects.
Put 'import __main__' after other idlelib imports.

Imports only needed for testing are put not at the top but in an
htest function def or "if __name__ == '__main__'" clause.

Within module imports like "from idlelib.mod import class" may cause
circular imports to deadlock.  Even without this, circular imports may
require at least one of the imports to be delayed until a function call.

What's New entries
------------------

Repository directory Doc/whatsnew/ has a file 3.n.rst for each 3.n
Python version.  For the first entry in each file, add subsection
'IDLE and idlelib', in alphabetical position, to the 'Improved Modules'
section.  For the rest of cpython, entries to 3.(n+1).rst begin with
the release of 3.n.0b1.  For IDLE, entries for features backported from
'main' to '3.n' during its beta period do not got in 3.(n+1).rst.  The
latter usually gets its first entry during the 3.n.0 candidate period
or after the 3.n.0 release.

When, as per PEP 434, feature changes are backported, entries are placed
in the 3.n.rst file *in the main branch* for each Python version n that
gets the backport. (Note: the format of entries have varied between
versions.)  Add a line "New in 3.n maintenance releases." before the
first back-ported feature after 3.n.0 is released. Since each older
version file gets a different number of backports, it is easiest to
make a separate PR for each file and label it with the backports
needed.

Github repository and issues
----------------------------

The CPython repository is https://github.com/python/cpython.  The
IDLE Issues listing is https://github.com/orgs/python/projects/31.
The main classification is by Topic, based on the IDLE menu.  View the
topics list by clicking the [<]] button in the upper right.
