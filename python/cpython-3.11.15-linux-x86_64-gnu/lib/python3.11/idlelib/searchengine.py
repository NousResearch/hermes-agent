'''Define SearchEngine for search dialogs.'''
import re
re.PatternError = re.error  # New in 3.13.

from tkinter import StringVar, BooleanVar, TclError
from tkinter import messagebox

def get(root):
    '''Return the singleton SearchEngine instance for the process.

    The single SearchEngine saves settings between dialog instances.
    If there is not a SearchEngine already, make one.
    '''
    if not hasattr(root, "_searchengine"):
        root._searchengine = SearchEngine(root)
        # This creates a cycle that persists until root is deleted.
    return root._searchengine


class SearchEngine:
    """Handles searching a text widget for Find, Replace, and Grep."""

    def __init__(self, root):
        '''Initialize Variables that save search state.

        The dialogs bind these to the UI elements present in the dialogs.
        '''
        self.root = root  # need for report_error()
        self.patvar = StringVar(root, '')   # search pattern
        self.revar = BooleanVar(root, False)   # regular expression?
        self.casevar = BooleanVar(root, False)   # match case?
        self.wordvar = BooleanVar(root, False)   # match whole word?
        self.wrapvar = BooleanVar(root, True)   # wrap around buffer?
        self.backvar = BooleanVar(root, False)   # search backwards?

    # Access methods

    def getpat(self):
        return self.patvar.get()

    def setpat(self, pat):
        self.patvar.set(pat)

    def isre(self):
        return self.revar.get()

    def iscase(self):
        return self.casevar.get()

    def isword(self):
        return self.wordvar.get()

    def iswrap(self):
        return self.wrapvar.get()

    def isback(self):
        return self.backvar.get()

    # Higher level access methods

    def setcookedpat(self, pat):
        "Set pattern after escaping if re."
        # called only in search.py: 66
        if self.isre():
            pat = re.escape(pat)
        self.setpat(pat)

    def getcookedpat(self):
        pat = self.getpat()
        if not self.isre():  # if True, see setcookedpat
            pat = re.escape(pat)
        if self.isword():
            pat = r"\b%s\b" % pat
        return pat

    def getprog(self):
        "Return compiled cooked search pattern."
        pat = self.getpat()
        if not pat:
            self.report_error(pat, "Empty regular expression")
            return None
        pat = self.getcookedpat()
        flags = 0
        if not self.iscase():
            flags = flags | re.IGNORECASE
        try:
            prog = re.compile(pat, flags)
        except re.PatternError as e:
            self.report_error(pat, e.msg, e.pos)
            return None
        return prog

    def report_error(self, pat, msg, col=None):
        # Derived class could override this with something fancier
        msg = "Error: " + str(msg)
        if pat:
            msg = msg + "\nPattern: " + str(pat)
        if col is not None:
            msg = msg + "\nOffset: " + str(col)
        messagebox.showerror("Regular expression error",
                               msg, master=self.root)

    def search_text(self, text, prog=None, ok=0):
        '''Return (lineno, matchobj) or None for forward/backward search.

        This function calls the right function with the right arguments.
        It directly return the result of that call.

        Text is a text widget. Prog is a precompiled pattern.
        The ok parameter is a bit complicated as it has two effects.

        If there is a selection, the search begin at either end,
        depending on the direction setting and ok, with ok meaning that
        the search starts with the selection. Otherwise, search begins
        at the insert mark.

        To aid progress, the search functions do not return an empty
        match at the starting position unless ok is True.
        '''

        if not prog:
            prog = self.getprog()
            if not prog:
                return None # Compilation failed -- stop
        wrap = self.wrapvar.get()
        first, last = get_selection(text)
        if self.isback():
            if ok:
                start = last
            else:
                start = first
            line, col = get_line_col(start)
            res = self.search_backward(text, prog, line, col, wrap, ok)
        else:
            if ok:
                start = first
            else:
                start = last
            line, col = get_line_col(start)
            res = self.search_forward(text, prog, line, col, wrap, ok)
        return res

    def search_forward(self, text, prog, line, col, wrap, ok=0):
        wrapped = 0
        startline = line
        chars = text.get("%d.0" % line, "%d.0" % (line+1))
        while chars:
            m = prog.search(chars[:-1], col)
            if m:
                if ok or m.end() > col:
                    return line, m
            line = line + 1
            if wrapped and line > startline:
                break
            col = 0
            ok = 1
            chars = text.get("%d.0" % line, "%d.0" % (line+1))
            if not chars and wrap:
                wrapped = 1
                wrap = 0
                line = 1
                chars = text.get("1.0", "2.0")
        return None

    def search_backward(self, text, prog, line, col, wrap, ok=0):
        wrapped = 0
        startline = line
        chars = text.get("%d.0" % line, "%d.0" % (line+1))
        while True:
            m = search_reverse(prog, chars[:-1], col)
            if m:
                if ok or m.start() < col:
                    return line, m
            line = line - 1
            if wrapped and line < startline:
                break
            ok = 1
            if line <= 0:
                if not wrap:
                    break
                wrapped = 1
                wrap = 0
                pos = text.index("end-1c")
                line, col = map(int, pos.split("."))
            chars = text.get("%d.0" % line, "%d.0" % (line+1))
            col = len(chars) - 1
        return None


def search_reverse(prog, chars, col):
    '''Search backwards and return an re match object or None.

    This is done by searching forwards until there is no match.
    Prog: compiled re object with a search method returning a match.
    Chars: line of text, without \\n.
    Col: stop index for the search; the limit for match.end().
    '''
    m = prog.search(chars)
    if not m:
        return None
    found = None
    i, j = m.span()  # m.start(), m.end() == match slice indexes
    while i < col and j <= col:
        found = m
        if i == j:
            j = j+1
        m = prog.search(chars, j)
        if not m:
            break
        i, j = m.span()
    return found

def get_selection(text):
    '''Return tuple of 'line.col' indexes from selection or insert mark.
    '''
    try:
        first = text.index("sel.first")
        last = text.index("sel.last")
    except TclError:
        first = last = None
    if not first:
        first = text.index("insert")
    if not last:
        last = first
    return first, last

def get_line_col(index):
    '''Return (line, col) tuple of ints from 'line.col' string.'''
    line, col = map(int, index.split(".")) # Fails on invalid index
    return line, col


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_searchengine', verbosity=2)
