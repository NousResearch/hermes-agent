# tk common message boxes
#
# this module provides an interface to the native message boxes
# available in Tk 4.2 and newer.
#
# written by Fredrik Lundh, May 1997
#

#
# options (all have default values):
#
# - default: which button to make default (one of the reply codes)
#
# - icon: which icon to display (see below)
#
# - message: the message to display
#
# - parent: which window to place the dialog on top of
#
# - title: dialog title
#
# - type: dialog type; that is, which buttons to display (see below)
#

from tkinter.commondialog import Dialog

__all__ = ["showinfo", "showwarning", "showerror",
           "askquestion", "askokcancel", "askyesno",
           "askyesnocancel", "askretrycancel"]

#
# constants

# icons
ERROR = "error"
INFO = "info"
QUESTION = "question"
WARNING = "warning"

# types
ABORTRETRYIGNORE = "abortretryignore"
OK = "ok"
OKCANCEL = "okcancel"
RETRYCANCEL = "retrycancel"
YESNO = "yesno"
YESNOCANCEL = "yesnocancel"

# replies
ABORT = "abort"
RETRY = "retry"
IGNORE = "ignore"
OK = "ok"
CANCEL = "cancel"
YES = "yes"
NO = "no"


#
# message dialog class

class Message(Dialog):
    "A message box"

    command  = "tk_messageBox"


#
# convenience stuff

# Rename _icon and _type options to allow overriding them in options
def _show(title=None, message=None, _icon=None, _type=None, **options):
    if _icon and "icon" not in options:    options["icon"] = _icon
    if _type and "type" not in options:    options["type"] = _type
    if title:   options["title"] = title
    if message: options["message"] = message
    res = Message(**options).show()
    # In some Tcl installations, yes/no is converted into a boolean.
    if isinstance(res, bool):
        if res:
            return YES
        return NO
    # In others we get a Tcl_Obj.
    return str(res)


def showinfo(title=None, message=None, **options):
    "Show an info message"
    return _show(title, message, INFO, OK, **options)


def showwarning(title=None, message=None, **options):
    "Show a warning message"
    return _show(title, message, WARNING, OK, **options)


def showerror(title=None, message=None, **options):
    "Show an error message"
    return _show(title, message, ERROR, OK, **options)


def askquestion(title=None, message=None, **options):
    "Ask a question"
    return _show(title, message, QUESTION, YESNO, **options)


def askokcancel(title=None, message=None, **options):
    "Ask if operation should proceed; return true if the answer is ok"
    s = _show(title, message, QUESTION, OKCANCEL, **options)
    return s == OK


def askyesno(title=None, message=None, **options):
    "Ask a question; return true if the answer is yes"
    s = _show(title, message, QUESTION, YESNO, **options)
    return s == YES


def askyesnocancel(title=None, message=None, **options):
    "Ask a question; return true if the answer is yes, None if cancelled."
    s = _show(title, message, QUESTION, YESNOCANCEL, **options)
    # s might be a Tcl index object, so convert it to a string
    s = str(s)
    if s == CANCEL:
        return None
    return s == YES


def askretrycancel(title=None, message=None, **options):
    "Ask if operation should be retried; return true if the answer is yes"
    s = _show(title, message, WARNING, RETRYCANCEL, **options)
    return s == RETRY


# --------------------------------------------------------------------
# test stuff

if __name__ == "__main__":

    print("info", showinfo("Spam", "Egg Information"))
    print("warning", showwarning("Spam", "Egg Warning"))
    print("error", showerror("Spam", "Egg Alert"))
    print("question", askquestion("Spam", "Question?"))
    print("proceed", askokcancel("Spam", "Proceed?"))
    print("yes/no", askyesno("Spam", "Got it?"))
    print("yes/no/cancel", askyesnocancel("Spam", "Want it?"))
    print("try again", askretrycancel("Spam", "Try again?"))
