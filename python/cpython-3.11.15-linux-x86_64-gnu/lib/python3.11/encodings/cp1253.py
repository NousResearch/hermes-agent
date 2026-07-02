""" Python Character Mapping Codec cp1253 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1253.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp1253',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> NULL
    '\x01'     #  0x01 -> START OF HEADING
    '\x02'     #  0x02 -> START OF TEXT
    '\x03'     #  0x03 -> END OF TEXT
    '\x04'     #  0x04 -> END OF TRANSMISSION
    '\x05'     #  0x05 -> ENQUIRY
    '\x06'     #  0x06 -> ACKNOWLEDGE
    '\x07'     #  0x07 -> BELL
    '\x08'     #  0x08 -> BACKSPACE
    '\t'       #  0x09 -> HORIZONTAL TABULATION
    '\n'       #  0x0A -> LINE FEED
    '\x0b'     #  0x0B -> VERTICAL TABULATION
    '\x0c'     #  0x0C -> FORM FEED
    '\r'       #  0x0D -> CARRIAGE RETURN
    '\x0e'     #  0x0E -> SHIFT OUT
    '\x0f'     #  0x0F -> SHIFT IN
    '\x10'     #  0x10 -> DATA LINK ESCAPE
    '\x11'     #  0x11 -> DEVICE CONTROL ONE
    '\x12'     #  0x12 -> DEVICE CONTROL TWO
    '\x13'     #  0x13 -> DEVICE CONTROL THREE
    '\x14'     #  0x14 -> DEVICE CONTROL FOUR
    '\x15'     #  0x15 -> NEGATIVE ACKNOWLEDGE
    '\x16'     #  0x16 -> SYNCHRONOUS IDLE
    '\x17'     #  0x17 -> END OF TRANSMISSION BLOCK
    '\x18'     #  0x18 -> CANCEL
    '\x19'     #  0x19 -> END OF MEDIUM
    '\x1a'     #  0x1A -> SUBSTITUTE
    '\x1b'     #  0x1B -> ESCAPE
    '\x1c'     #  0x1C -> FILE SEPARATOR
    '\x1d'     #  0x1D -> GROUP SEPARATOR
    '\x1e'     #  0x1E -> RECORD SEPARATOR
    '\x1f'     #  0x1F -> UNIT SEPARATOR
    ' '        #  0x20 -> SPACE
    '!'        #  0x21 -> EXCLAMATION MARK
    '"'        #  0x22 -> QUOTATION MARK
    '#'        #  0x23 -> NUMBER SIGN
    '$'        #  0x24 -> DOLLAR SIGN
    '%'        #  0x25 -> PERCENT SIGN
    '&'        #  0x26 -> AMPERSAND
    "'"        #  0x27 -> APOSTROPHE
    '('        #  0x28 -> LEFT PARENTHESIS
    ')'        #  0x29 -> RIGHT PARENTHESIS
    '*'        #  0x2A -> ASTERISK
    '+'        #  0x2B -> PLUS SIGN
    ','        #  0x2C -> COMMA
    '-'        #  0x2D -> HYPHEN-MINUS
    '.'        #  0x2E -> FULL STOP
    '/'        #  0x2F -> SOLIDUS
    '0'        #  0x30 -> DIGIT ZERO
    '1'        #  0x31 -> DIGIT ONE
    '2'        #  0x32 -> DIGIT TWO
    '3'        #  0x33 -> DIGIT THREE
    '4'        #  0x34 -> DIGIT FOUR
    '5'        #  0x35 -> DIGIT FIVE
    '6'        #  0x36 -> DIGIT SIX
    '7'        #  0x37 -> DIGIT SEVEN
    '8'        #  0x38 -> DIGIT EIGHT
    '9'        #  0x39 -> DIGIT NINE
    ':'        #  0x3A -> COLON
    ';'        #  0x3B -> SEMICOLON
    '<'        #  0x3C -> LESS-THAN SIGN
    '='        #  0x3D -> EQUALS SIGN
    '>'        #  0x3E -> GREATER-THAN SIGN
    '?'        #  0x3F -> QUESTION MARK
    '@'        #  0x40 -> COMMERCIAL AT
    'A'        #  0x41 -> LATIN CAPITAL LETTER A
    'B'        #  0x42 -> LATIN CAPITAL LETTER B
    'C'        #  0x43 -> LATIN CAPITAL LETTER C
    'D'        #  0x44 -> LATIN CAPITAL LETTER D
    'E'        #  0x45 -> LATIN CAPITAL LETTER E
    'F'        #  0x46 -> LATIN CAPITAL LETTER F
    'G'        #  0x47 -> LATIN CAPITAL LETTER G
    'H'        #  0x48 -> LATIN CAPITAL LETTER H
    'I'        #  0x49 -> LATIN CAPITAL LETTER I
    'J'        #  0x4A -> LATIN CAPITAL LETTER J
    'K'        #  0x4B -> LATIN CAPITAL LETTER K
    'L'        #  0x4C -> LATIN CAPITAL LETTER L
    'M'        #  0x4D -> LATIN CAPITAL LETTER M
    'N'        #  0x4E -> LATIN CAPITAL LETTER N
    'O'        #  0x4F -> LATIN CAPITAL LETTER O
    'P'        #  0x50 -> LATIN CAPITAL LETTER P
    'Q'        #  0x51 -> LATIN CAPITAL LETTER Q
    'R'        #  0x52 -> LATIN CAPITAL LETTER R
    'S'        #  0x53 -> LATIN CAPITAL LETTER S
    'T'        #  0x54 -> LATIN CAPITAL LETTER T
    'U'        #  0x55 -> LATIN CAPITAL LETTER U
    'V'        #  0x56 -> LATIN CAPITAL LETTER V
    'W'        #  0x57 -> LATIN CAPITAL LETTER W
    'X'        #  0x58 -> LATIN CAPITAL LETTER X
    'Y'        #  0x59 -> LATIN CAPITAL LETTER Y
    'Z'        #  0x5A -> LATIN CAPITAL LETTER Z
    '['        #  0x5B -> LEFT SQUARE BRACKET
    '\\'       #  0x5C -> REVERSE SOLIDUS
    ']'        #  0x5D -> RIGHT SQUARE BRACKET
    '^'        #  0x5E -> CIRCUMFLEX ACCENT
    '_'        #  0x5F -> LOW LINE
    '`'        #  0x60 -> GRAVE ACCENT
    'a'        #  0x61 -> LATIN SMALL LETTER A
    'b'        #  0x62 -> LATIN SMALL LETTER B
    'c'        #  0x63 -> LATIN SMALL LETTER C
    'd'        #  0x64 -> LATIN SMALL LETTER D
    'e'        #  0x65 -> LATIN SMALL LETTER E
    'f'        #  0x66 -> LATIN SMALL LETTER F
    'g'        #  0x67 -> LATIN SMALL LETTER G
    'h'        #  0x68 -> LATIN SMALL LETTER H
    'i'        #  0x69 -> LATIN SMALL LETTER I
    'j'        #  0x6A -> LATIN SMALL LETTER J
    'k'        #  0x6B -> LATIN SMALL LETTER K
    'l'        #  0x6C -> LATIN SMALL LETTER L
    'm'        #  0x6D -> LATIN SMALL LETTER M
    'n'        #  0x6E -> LATIN SMALL LETTER N
    'o'        #  0x6F -> LATIN SMALL LETTER O
    'p'        #  0x70 -> LATIN SMALL LETTER P
    'q'        #  0x71 -> LATIN SMALL LETTER Q
    'r'        #  0x72 -> LATIN SMALL LETTER R
    's'        #  0x73 -> LATIN SMALL LETTER S
    't'        #  0x74 -> LATIN SMALL LETTER T
    'u'        #  0x75 -> LATIN SMALL LETTER U
    'v'        #  0x76 -> LATIN SMALL LETTER V
    'w'        #  0x77 -> LATIN SMALL LETTER W
    'x'        #  0x78 -> LATIN SMALL LETTER X
    'y'        #  0x79 -> LATIN SMALL LETTER Y
    'z'        #  0x7A -> LATIN SMALL LETTER Z
    '{'        #  0x7B -> LEFT CURLY BRACKET
    '|'        #  0x7C -> VERTICAL LINE
    '}'        #  0x7D -> RIGHT CURLY BRACKET
    '~'        #  0x7E -> TILDE
    '\x7f'     #  0x7F -> DELETE
    '\u20ac'   #  0x80 -> EURO SIGN
    '\ufffe'   #  0x81 -> UNDEFINED
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\ufffe'   #  0x88 -> UNDEFINED
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\ufffe'   #  0x8A -> UNDEFINED
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x8C -> UNDEFINED
    '\ufffe'   #  0x8D -> UNDEFINED
    '\ufffe'   #  0x8E -> UNDEFINED
    '\ufffe'   #  0x8F -> UNDEFINED
    '\ufffe'   #  0x90 -> UNDEFINED
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0385'   #  0xA1 -> GREEK DIALYTIKA TONOS
    '\u0386'   #  0xA2 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\ufffe'   #  0xAA -> UNDEFINED
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u2015'   #  0xAF -> HORIZONTAL BAR
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\u0384'   #  0xB4 -> GREEK TONOS
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u0388'   #  0xB8 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0389'   #  0xB9 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\u038a'   #  0xBA -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u038c'   #  0xBC -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\u038e'   #  0xBE -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u038f'   #  0xBF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '\u0390'   #  0xC0 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u0391'   #  0xC1 -> GREEK CAPITAL LETTER ALPHA
    '\u0392'   #  0xC2 -> GREEK CAPITAL LETTER BETA
    '\u0393'   #  0xC3 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0xC4 -> GREEK CAPITAL LETTER DELTA
    '\u0395'   #  0xC5 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0xC6 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0xC7 -> GREEK CAPITAL LETTER ETA
    '\u0398'   #  0xC8 -> GREEK CAPITAL LETTER THETA
    '\u0399'   #  0xC9 -> GREEK CAPITAL LETTER IOTA
    '\u039a'   #  0xCA -> GREEK CAPITAL LETTER KAPPA
    '\u039b'   #  0xCB -> GREEK CAPITAL LETTER LAMDA
    '\u039c'   #  0xCC -> GREEK CAPITAL LETTER MU
    '\u039d'   #  0xCD -> GREEK CAPITAL LETTER NU
    '\u039e'   #  0xCE -> GREEK CAPITAL LETTER XI
    '\u039f'   #  0xCF -> GREEK CAPITAL LETTER OMICRON
    '\u03a0'   #  0xD0 -> GREEK CAPITAL LETTER PI
    '\u03a1'   #  0xD1 -> GREEK CAPITAL LETTER RHO
    '\ufffe'   #  0xD2 -> UNDEFINED
    '\u03a3'   #  0xD3 -> GREEK CAPITAL LETTER SIGMA
    '\u03a4'   #  0xD4 -> GREEK CAPITAL LETTER TAU
    '\u03a5'   #  0xD5 -> GREEK CAPITAL LETTER UPSILON
    '\u03a6'   #  0xD6 -> GREEK CAPITAL LETTER PHI
    '\u03a7'   #  0xD7 -> GREEK CAPITAL LETTER CHI
    '\u03a8'   #  0xD8 -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0xD9 -> GREEK CAPITAL LETTER OMEGA
    '\u03aa'   #  0xDA -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\u03ab'   #  0xDB -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '\u03ac'   #  0xDC -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u03ad'   #  0xDD -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xDE -> GREEK SMALL LETTER ETA WITH TONOS
    '\u03af'   #  0xDF -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03b0'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    '\u03b3'   #  0xE3 -> GREEK SMALL LETTER GAMMA
    '\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    '\u03b6'   #  0xE6 -> GREEK SMALL LETTER ZETA
    '\u03b7'   #  0xE7 -> GREEK SMALL LETTER ETA
    '\u03b8'   #  0xE8 -> GREEK SMALL LETTER THETA
    '\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    '\u03ba'   #  0xEA -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0xEB -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0xEC -> GREEK SMALL LETTER MU
    '\u03bd'   #  0xED -> GREEK SMALL LETTER NU
    '\u03be'   #  0xEE -> GREEK SMALL LETTER XI
    '\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    '\u03c1'   #  0xF1 -> GREEK SMALL LETTER RHO
    '\u03c2'   #  0xF2 -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    '\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    '\u03c5'   #  0xF5 -> GREEK SMALL LETTER UPSILON
    '\u03c6'   #  0xF6 -> GREEK SMALL LETTER PHI
    '\u03c7'   #  0xF7 -> GREEK SMALL LETTER CHI
    '\u03c8'   #  0xF8 -> GREEK SMALL LETTER PSI
    '\u03c9'   #  0xF9 -> GREEK SMALL LETTER OMEGA
    '\u03ca'   #  0xFA -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03cb'   #  0xFB -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u03cc'   #  0xFC -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u03cd'   #  0xFD -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03ce'   #  0xFE -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
