Original IDLE todo, much of it now outdated:
============================================
TO DO:

- improve debugger:
    - manage breakpoints globally, allow bp deletion, tbreak, cbreak etc.
    - real object browser
    - help on how to use it (a simple help button will do wonders)
    - performance?  (updates of large sets of locals are slow)
    - better integration of "debug module"
    - debugger should be global resource (attached to flist, not to shell)
    - fix the stupid bug where you need to step twice
    - display class name in stack viewer entries for methods
    - suppress tracing through IDLE internals (e.g. print) DONE
    - add a button to suppress through a specific module or class or method
    - more object inspection to stack viewer, e.g. to view all array items
- insert the initial current directory into sys.path DONE
- default directory attribute for each window instead of only for windows
  that have an associated filename
- command expansion from keywords, module contents, other buffers, etc.
- "Recent documents" menu item DONE
- Filter region command
- Optional horizontal scroll bar
- more Emacsisms:
    - ^K should cut to buffer
    - M-[, M-] to move by paragraphs
    - incremental search?
- search should indicate wrap-around in some way
- restructure state sensitive code to avoid testing flags all the time
- persistent user state (e.g. window and cursor positions, bindings)
- make backups when saving
- check file mtimes at various points
- Pluggable interface with RCS/CVS/Perforce/Clearcase
- better help?
- don't open second class browser on same module (nor second path browser)
- unify class and path browsers
- Need to define a standard way whereby one can determine one is running
  inside IDLE (needed for Tk mainloop, also handy for $PYTHONSTARTUP)
- Add more utility methods for use by extensions (a la get_selection)
- Way to run command in totally separate interpreter (fork+os.system?) DONE
- Way to find definition of fully-qualified name:
  In other words, select "UserDict.UserDict", hit some magic key and
  it loads up UserDict.py and finds the first def or class for UserDict.
- need a way to force colorization on/off
- need a way to force auto-indent on/off

Details:

- ^O (on Unix -- open-line) should honor autoindent
- after paste, show end of pasted text
- on Windows, should turn short filename to long filename (not only in argv!)
  (shouldn't this be done -- or undone -- by ntpath.normpath?)
- new autoindent after colon even indents when the colon is in a comment!
- sometimes forward slashes in pathname remain
- sometimes star in window name remains in Windows menu
- With unix bindings, ESC by itself is ignored
- Sometimes for no apparent reason a selection from the cursor to the
  end of the command buffer appears, which is hard to get rid of
  because it stays when you are typing!
- The Line/Col in the status bar can be wrong initially in PyShell DONE

Structural problems:

- too much knowledge in FileList about EditorWindow (for example)
- should add some primitives for accessing the selection etc.
  to repeat cumbersome code over and over

======================================================================

Jeff Bauer suggests:

- Open Module doesn't appear to handle hierarchical packages.
- Class browser should also allow hierarchical packages.
- Open and Open Module could benefit from a history, DONE
  either command line style, or Microsoft recent-file
  style.
- Add a Smalltalk-style inspector  (i.e. Tkinspect)

The last suggestion is already a reality, but not yet
integrated into IDLE.  I use a module called inspector.py,
that used to be available from python.org(?)  It no longer
appears to be in the contributed section, and the source
has no author attribution.

In any case, the code is useful for visually navigating
an object's attributes, including its container hierarchy.

    >>> from inspector import Tkinspect
    >>> Tkinspect(None, myObject)

Tkinspect could probably be extended and refined to
integrate better into IDLE.

======================================================================

Comparison to PTUI
------------------

+ PTUI's help is better (HTML!)

+ PTUI can attach a shell to any module

+ PTUI has some more I/O commands:
  open multiple
  append
  examine (what's that?)

======================================================================

Notes after trying to run Grail
-------------------------------

- Grail does stuff to sys.path based on sys.argv[0]; you must set
sys.argv[0] to something decent first (it is normally set to the path of
the idle script).

- Grail must be exec'ed in __main__ because that's imported by some
other parts of Grail.

- Grail uses a module called History and so does idle :-(

======================================================================

Robin Friedrich's items:

Things I'd like to see:
    - I'd like support for shift-click extending the selection. There's a
      bug now that it doesn't work the first time you try it.
    - Printing is needed. How hard can that be on Windows? FIRST CUT DONE
    - The python-mode trick of autoindenting a line with <tab> is neat and
      very handy.
    - (someday) a spellchecker for docstrings and comments.
    - a pagedown/up command key which moves to next class/def statement (top
      level)
    - split window capability
    - DnD text relocation/copying

Things I don't want to see.
    - line numbers...  will probably slow things down way too much.
    - Please use another icon for the tree browser leaf. The small snake
      isn't cutting it.

----------------------------------------------------------------------

- Customizable views (multi-window or multi-pane).  (Markus Gritsch)

- Being able to double click (maybe double right click) on a callable
object in the editor which shows the source of the object, if
possible.  (Gerrit Holl)

- Hooks into the guts, like in Emacs.  (Mike Romberg)

- Sharing the editor with a remote tutor.  (Martijn Faassen)

- Multiple views on the same file.  (Tony J Ibbs)

- Store breakpoints in a global (per-project) database (GvR); Dirk
Heise adds: save some space-trimmed context and search around when
reopening a file that might have been edited by someone else.

- Capture menu events in extensions without changing the IDLE source.
(Matthias Barmeier)

- Use overlapping panels (a "notebook" in MFC terms I think) for info
that doesn't need to be accessible simultaneously (e.g. HTML source
and output).  Use multi-pane windows for info that does need to be
shown together (e.g. class browser and source).  (Albert Brandl)

- A project should invisibly track all symbols, for instant search,
replace and cross-ref.  Projects should be allowed to span multiple
directories, hosts, etc.  Project management files are placed in a
directory you specify.  A global mapping between project names and
project directories should exist [not so sure --GvR].  (Tim Peters)

- Merge attr-tips and auto-expand.  (Mark Hammond, Tim Peters)

- Python Shell should behave more like a "shell window" as users know
it -- i.e. you can only edit the current command, and the cursor can't
escape from the command area.  (Albert Brandl)

- Set X11 class to "idle/Idle", set icon and title to something
beginning with "idle" -- for window manangers.  (Randall Hopper)

- Config files editable through a preferences dialog.  (me) DONE

- Config files still editable outside the preferences dialog.
(Randall Hopper) DONE

- When you're editing a command in PyShell, and there are only blank
lines below the cursor, hitting Return should ignore or delete those
blank lines rather than deciding you're not on the last line.  (me)

- Run command (F5 c.s.) should be more like Pythonwin's Run -- a
dialog with options to give command line arguments, run the debugger,
etc.  (me)

- Shouldn't be able to delete part of the prompt (or any text before
it) in the PyShell.  (Martijn Faassen)   DONE

- Emacs style auto-fill (also smart about comments and strings).
(Jeremy Hylton)

- Output of Run Script should go to a separate output window, not to
the shell window.  Output of separate runs should all go to the same
window but clearly delimited.  (David Scherer) REJECT FIRST, LATTER DONE

- GUI form designer to kick VB's butt.  (Robert Geiger) THAT'S NOT IDLE

- Printing!  Possibly via generation of PDF files which the user must
then send to the printer separately.  (Dinu Gherman)  FIRST CUT
