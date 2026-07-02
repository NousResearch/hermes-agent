"""Constants and membership tests for ASCII characters"""

NUL     = 0x00  # ^@
SOH     = 0x01  # ^A
STX     = 0x02  # ^B
ETX     = 0x03  # ^C
EOT     = 0x04  # ^D
ENQ     = 0x05  # ^E
ACK     = 0x06  # ^F
BEL     = 0x07  # ^G
BS      = 0x08  # ^H
TAB     = 0x09  # ^I
HT      = 0x09  # ^I
LF      = 0x0a  # ^J
NL      = 0x0a  # ^J
VT      = 0x0b  # ^K
FF      = 0x0c  # ^L
CR      = 0x0d  # ^M
SO      = 0x0e  # ^N
SI      = 0x0f  # ^O
DLE     = 0x10  # ^P
DC1     = 0x11  # ^Q
DC2     = 0x12  # ^R
DC3     = 0x13  # ^S
DC4     = 0x14  # ^T
NAK     = 0x15  # ^U
SYN     = 0x16  # ^V
ETB     = 0x17  # ^W
CAN     = 0x18  # ^X
EM      = 0x19  # ^Y
SUB     = 0x1a  # ^Z
ESC     = 0x1b  # ^[
FS      = 0x1c  # ^\
GS      = 0x1d  # ^]
RS      = 0x1e  # ^^
US      = 0x1f  # ^_
SP      = 0x20  # space
DEL     = 0x7f  # delete

controlnames = [
"NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL",
"BS",  "HT",  "LF",  "VT",  "FF",  "CR",  "SO",  "SI",
"DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB",
"CAN", "EM",  "SUB", "ESC", "FS",  "GS",  "RS",  "US",
"SP"
]

def _ctoi(c):
    if type(c) == type(""):
        return ord(c)
    else:
        return c

def isalnum(c): return isalpha(c) or isdigit(c)
def isalpha(c): return isupper(c) or islower(c)
def isascii(c): return 0 <= _ctoi(c) <= 127          # ?
def isblank(c): return _ctoi(c) in (9, 32)
def iscntrl(c): return 0 <= _ctoi(c) <= 31 or _ctoi(c) == 127
def isdigit(c): return 48 <= _ctoi(c) <= 57
def isgraph(c): return 33 <= _ctoi(c) <= 126
def islower(c): return 97 <= _ctoi(c) <= 122
def isprint(c): return 32 <= _ctoi(c) <= 126
def ispunct(c): return isgraph(c) and not isalnum(c)
def isspace(c): return _ctoi(c) in (9, 10, 11, 12, 13, 32)
def isupper(c): return 65 <= _ctoi(c) <= 90
def isxdigit(c): return isdigit(c) or \
    (65 <= _ctoi(c) <= 70) or (97 <= _ctoi(c) <= 102)
def isctrl(c): return 0 <= _ctoi(c) < 32
def ismeta(c): return _ctoi(c) > 127

def ascii(c):
    if type(c) == type(""):
        return chr(_ctoi(c) & 0x7f)
    else:
        return _ctoi(c) & 0x7f

def ctrl(c):
    if type(c) == type(""):
        return chr(_ctoi(c) & 0x1f)
    else:
        return _ctoi(c) & 0x1f

def alt(c):
    if type(c) == type(""):
        return chr(_ctoi(c) | 0x80)
    else:
        return _ctoi(c) | 0x80

def unctrl(c):
    bits = _ctoi(c)
    if bits == 0x7f:
        rep = "^?"
    elif isprint(bits & 0x7f):
        rep = chr(bits & 0x7f)
    else:
        rep = "^" + chr(((bits & 0x7f) | 0x20) + 0x20)
    if bits & 0x80:
        return "!" + rep
    return rep
