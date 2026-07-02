"""Mailcap file handling.  See RFC 1524."""

import os
import warnings
import re

__all__ = ["getcaps","findmatch"]


_DEPRECATION_MSG = ('The {name} module is deprecated and will be removed in '
                    'Python {remove}. See the mimetypes module for an '
                    'alternative.')
warnings._deprecated(__name__, _DEPRECATION_MSG, remove=(3, 13))


def lineno_sort_key(entry):
    # Sort in ascending order, with unspecified entries at the end
    if 'lineno' in entry:
        return 0, entry['lineno']
    else:
        return 1, 0

_find_unsafe = re.compile(r'[^\xa1-\U0010FFFF\w@+=:,./-]').search

class UnsafeMailcapInput(Warning):
    """Warning raised when refusing unsafe input"""


# Part 1: top-level interface.

def getcaps():
    """Return a dictionary containing the mailcap database.

    The dictionary maps a MIME type (in all lowercase, e.g. 'text/plain')
    to a list of dictionaries corresponding to mailcap entries.  The list
    collects all the entries for that MIME type from all available mailcap
    files.  Each dictionary contains key-value pairs for that MIME type,
    where the viewing command is stored with the key "view".

    """
    caps = {}
    lineno = 0
    for mailcap in listmailcapfiles():
        try:
            fp = open(mailcap, 'r')
        except OSError:
            continue
        with fp:
            morecaps, lineno = _readmailcapfile(fp, lineno)
        for key, value in morecaps.items():
            if not key in caps:
                caps[key] = value
            else:
                caps[key] = caps[key] + value
    return caps

def listmailcapfiles():
    """Return a list of all mailcap files found on the system."""
    # This is mostly a Unix thing, but we use the OS path separator anyway
    if 'MAILCAPS' in os.environ:
        pathstr = os.environ['MAILCAPS']
        mailcaps = pathstr.split(os.pathsep)
    else:
        if 'HOME' in os.environ:
            home = os.environ['HOME']
        else:
            # Don't bother with getpwuid()
            home = '.' # Last resort
        mailcaps = [home + '/.mailcap', '/etc/mailcap',
                '/usr/etc/mailcap', '/usr/local/etc/mailcap']
    return mailcaps


# Part 2: the parser.
def readmailcapfile(fp):
    """Read a mailcap file and return a dictionary keyed by MIME type."""
    warnings.warn('readmailcapfile is deprecated, use getcaps instead',
                  DeprecationWarning, 2)
    caps, _ = _readmailcapfile(fp, None)
    return caps


def _readmailcapfile(fp, lineno):
    """Read a mailcap file and return a dictionary keyed by MIME type.

    Each MIME type is mapped to an entry consisting of a list of
    dictionaries; the list will contain more than one such dictionary
    if a given MIME type appears more than once in the mailcap file.
    Each dictionary contains key-value pairs for that MIME type, where
    the viewing command is stored with the key "view".
    """
    caps = {}
    while 1:
        line = fp.readline()
        if not line: break
        # Ignore comments and blank lines
        if line[0] == '#' or line.strip() == '':
            continue
        nextline = line
        # Join continuation lines
        while nextline[-2:] == '\\\n':
            nextline = fp.readline()
            if not nextline: nextline = '\n'
            line = line[:-2] + nextline
        # Parse the line
        key, fields = parseline(line)
        if not (key and fields):
            continue
        if lineno is not None:
            fields['lineno'] = lineno
            lineno += 1
        # Normalize the key
        types = key.split('/')
        for j in range(len(types)):
            types[j] = types[j].strip()
        key = '/'.join(types).lower()
        # Update the database
        if key in caps:
            caps[key].append(fields)
        else:
            caps[key] = [fields]
    return caps, lineno

def parseline(line):
    """Parse one entry in a mailcap file and return a dictionary.

    The viewing command is stored as the value with the key "view",
    and the rest of the fields produce key-value pairs in the dict.
    """
    fields = []
    i, n = 0, len(line)
    while i < n:
        field, i = parsefield(line, i, n)
        fields.append(field)
        i = i+1 # Skip semicolon
    if len(fields) < 2:
        return None, None
    key, view, rest = fields[0], fields[1], fields[2:]
    fields = {'view': view}
    for field in rest:
        i = field.find('=')
        if i < 0:
            fkey = field
            fvalue = ""
        else:
            fkey = field[:i].strip()
            fvalue = field[i+1:].strip()
        if fkey in fields:
            # Ignore it
            pass
        else:
            fields[fkey] = fvalue
    return key, fields

def parsefield(line, i, n):
    """Separate one key-value pair in a mailcap entry."""
    start = i
    while i < n:
        c = line[i]
        if c == ';':
            break
        elif c == '\\':
            i = i+2
        else:
            i = i+1
    return line[start:i].strip(), i


# Part 3: using the database.

def findmatch(caps, MIMEtype, key='view', filename="/dev/null", plist=[]):
    """Find a match for a mailcap entry.

    Return a tuple containing the command line, and the mailcap entry
    used; (None, None) if no match is found.  This may invoke the
    'test' command of several matching entries before deciding which
    entry to use.

    """
    if _find_unsafe(filename):
        msg = "Refusing to use mailcap with filename %r. Use a safe temporary filename." % (filename,)
        warnings.warn(msg, UnsafeMailcapInput)
        return None, None
    entries = lookup(caps, MIMEtype, key)
    # XXX This code should somehow check for the needsterminal flag.
    for e in entries:
        if 'test' in e:
            test = subst(e['test'], filename, plist)
            if test is None:
                continue
            if test and os.system(test) != 0:
                continue
        command = subst(e[key], MIMEtype, filename, plist)
        if command is not None:
            return command, e
    return None, None

def lookup(caps, MIMEtype, key=None):
    entries = []
    if MIMEtype in caps:
        entries = entries + caps[MIMEtype]
    MIMEtypes = MIMEtype.split('/')
    MIMEtype = MIMEtypes[0] + '/*'
    if MIMEtype in caps:
        entries = entries + caps[MIMEtype]
    if key is not None:
        entries = [e for e in entries if key in e]
    entries = sorted(entries, key=lineno_sort_key)
    return entries

def subst(field, MIMEtype, filename, plist=[]):
    # XXX Actually, this is Unix-specific
    res = ''
    i, n = 0, len(field)
    while i < n:
        c = field[i]; i = i+1
        if c != '%':
            if c == '\\':
                c = field[i:i+1]; i = i+1
            res = res + c
        else:
            c = field[i]; i = i+1
            if c == '%':
                res = res + c
            elif c == 's':
                res = res + filename
            elif c == 't':
                if _find_unsafe(MIMEtype):
                    msg = "Refusing to substitute MIME type %r into a shell command." % (MIMEtype,)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + MIMEtype
            elif c == '{':
                start = i
                while i < n and field[i] != '}':
                    i = i+1
                name = field[start:i]
                i = i+1
                param = findparam(name, plist)
                if _find_unsafe(param):
                    msg = "Refusing to substitute parameter %r (%s) into a shell command" % (param, name)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + param
            # XXX To do:
            # %n == number of parts if type is multipart/*
            # %F == list of alternating type and filename for parts
            else:
                res = res + '%' + c
    return res

def findparam(name, plist):
    name = name.lower() + '='
    n = len(name)
    for p in plist:
        if p[:n].lower() == name:
            return p[n:]
    return ''


# Part 4: test program.

def test():
    import sys
    caps = getcaps()
    if not sys.argv[1:]:
        show(caps)
        return
    for i in range(1, len(sys.argv), 2):
        args = sys.argv[i:i+2]
        if len(args) < 2:
            print("usage: mailcap [MIMEtype file] ...")
            return
        MIMEtype = args[0]
        file = args[1]
        command, e = findmatch(caps, MIMEtype, 'view', file)
        if not command:
            print("No viewer found for", type)
        else:
            print("Executing:", command)
            sts = os.system(command)
            sts = os.waitstatus_to_exitcode(sts)
            if sts:
                print("Exit status:", sts)

def show(caps):
    print("Mailcap files:")
    for fn in listmailcapfiles(): print("\t" + fn)
    print()
    if not caps: caps = getcaps()
    print("Mailcap entries:")
    print()
    ckeys = sorted(caps)
    for type in ckeys:
        print(type)
        entries = caps[type]
        for e in entries:
            keys = sorted(e)
            for k in keys:
                print("  %-15s" % k, e[k])
            print()

if __name__ == '__main__':
    test()
