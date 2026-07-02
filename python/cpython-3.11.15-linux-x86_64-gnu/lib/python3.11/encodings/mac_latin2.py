""" Python Character Mapping Codec mac_latin2 generated from 'MAPPINGS/VENDORS/MICSFT/MAC/LATIN2.TXT' with gencodec.py.

Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.
(c) Copyright 2000 Guido van Rossum.

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
        name='mac-latin2',
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
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u0100'   #  0x81 -> LATIN CAPITAL LETTER A WITH MACRON
    '\u0101'   #  0x82 -> LATIN SMALL LETTER A WITH MACRON
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0104'   #  0x84 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\u0105'   #  0x88 -> LATIN SMALL LETTER A WITH OGONEK
    '\u010c'   #  0x89 -> LATIN CAPITAL LETTER C WITH CARON
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u010d'   #  0x8B -> LATIN SMALL LETTER C WITH CARON
    '\u0106'   #  0x8C -> LATIN CAPITAL LETTER C WITH ACUTE
    '\u0107'   #  0x8D -> LATIN SMALL LETTER C WITH ACUTE
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\u0179'   #  0x8F -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\u017a'   #  0x90 -> LATIN SMALL LETTER Z WITH ACUTE
    '\u010e'   #  0x91 -> LATIN CAPITAL LETTER D WITH CARON
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\u010f'   #  0x93 -> LATIN SMALL LETTER D WITH CARON
    '\u0112'   #  0x94 -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0113'   #  0x95 -> LATIN SMALL LETTER E WITH MACRON
    '\u0116'   #  0x96 -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\u0117'   #  0x98 -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf5'     #  0x9B -> LATIN SMALL LETTER O WITH TILDE
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\u011a'   #  0x9D -> LATIN CAPITAL LETTER E WITH CARON
    '\u011b'   #  0x9E -> LATIN SMALL LETTER E WITH CARON
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u2020'   #  0xA0 -> DAGGER
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\u0118'   #  0xA2 -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u2122'   #  0xAA -> TRADE MARK SIGN
    '\u0119'   #  0xAB -> LATIN SMALL LETTER E WITH OGONEK
    '\xa8'     #  0xAC -> DIAERESIS
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\u0123'   #  0xAE -> LATIN SMALL LETTER G WITH CEDILLA
    '\u012e'   #  0xAF -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u012f'   #  0xB0 -> LATIN SMALL LETTER I WITH OGONEK
    '\u012a'   #  0xB1 -> LATIN CAPITAL LETTER I WITH MACRON
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\u012b'   #  0xB4 -> LATIN SMALL LETTER I WITH MACRON
    '\u0136'   #  0xB5 -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\u2202'   #  0xB6 -> PARTIAL DIFFERENTIAL
    '\u2211'   #  0xB7 -> N-ARY SUMMATION
    '\u0142'   #  0xB8 -> LATIN SMALL LETTER L WITH STROKE
    '\u013b'   #  0xB9 -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\u013c'   #  0xBA -> LATIN SMALL LETTER L WITH CEDILLA
    '\u013d'   #  0xBB -> LATIN CAPITAL LETTER L WITH CARON
    '\u013e'   #  0xBC -> LATIN SMALL LETTER L WITH CARON
    '\u0139'   #  0xBD -> LATIN CAPITAL LETTER L WITH ACUTE
    '\u013a'   #  0xBE -> LATIN SMALL LETTER L WITH ACUTE
    '\u0145'   #  0xBF -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\u0146'   #  0xC0 -> LATIN SMALL LETTER N WITH CEDILLA
    '\u0143'   #  0xC1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\xac'     #  0xC2 -> NOT SIGN
    '\u221a'   #  0xC3 -> SQUARE ROOT
    '\u0144'   #  0xC4 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0147'   #  0xC5 -> LATIN CAPITAL LETTER N WITH CARON
    '\u2206'   #  0xC6 -> INCREMENT
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xC8 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\u0148'   #  0xCB -> LATIN SMALL LETTER N WITH CARON
    '\u0150'   #  0xCC -> LATIN CAPITAL LETTER O WITH DOUBLE ACUTE
    '\xd5'     #  0xCD -> LATIN CAPITAL LETTER O WITH TILDE
    '\u0151'   #  0xCE -> LATIN SMALL LETTER O WITH DOUBLE ACUTE
    '\u014c'   #  0xCF -> LATIN CAPITAL LETTER O WITH MACRON
    '\u2013'   #  0xD0 -> EN DASH
    '\u2014'   #  0xD1 -> EM DASH
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u25ca'   #  0xD7 -> LOZENGE
    '\u014d'   #  0xD8 -> LATIN SMALL LETTER O WITH MACRON
    '\u0154'   #  0xD9 -> LATIN CAPITAL LETTER R WITH ACUTE
    '\u0155'   #  0xDA -> LATIN SMALL LETTER R WITH ACUTE
    '\u0158'   #  0xDB -> LATIN CAPITAL LETTER R WITH CARON
    '\u2039'   #  0xDC -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u203a'   #  0xDD -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\u0159'   #  0xDE -> LATIN SMALL LETTER R WITH CARON
    '\u0156'   #  0xDF -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\u0157'   #  0xE0 -> LATIN SMALL LETTER R WITH CEDILLA
    '\u0160'   #  0xE1 -> LATIN CAPITAL LETTER S WITH CARON
    '\u201a'   #  0xE2 -> SINGLE LOW-9 QUOTATION MARK
    '\u201e'   #  0xE3 -> DOUBLE LOW-9 QUOTATION MARK
    '\u0161'   #  0xE4 -> LATIN SMALL LETTER S WITH CARON
    '\u015a'   #  0xE5 -> LATIN CAPITAL LETTER S WITH ACUTE
    '\u015b'   #  0xE6 -> LATIN SMALL LETTER S WITH ACUTE
    '\xc1'     #  0xE7 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\u0164'   #  0xE8 -> LATIN CAPITAL LETTER T WITH CARON
    '\u0165'   #  0xE9 -> LATIN SMALL LETTER T WITH CARON
    '\xcd'     #  0xEA -> LATIN CAPITAL LETTER I WITH ACUTE
    '\u017d'   #  0xEB -> LATIN CAPITAL LETTER Z WITH CARON
    '\u017e'   #  0xEC -> LATIN SMALL LETTER Z WITH CARON
    '\u016a'   #  0xED -> LATIN CAPITAL LETTER U WITH MACRON
    '\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xEF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u016b'   #  0xF0 -> LATIN SMALL LETTER U WITH MACRON
    '\u016e'   #  0xF1 -> LATIN CAPITAL LETTER U WITH RING ABOVE
    '\xda'     #  0xF2 -> LATIN CAPITAL LETTER U WITH ACUTE
    '\u016f'   #  0xF3 -> LATIN SMALL LETTER U WITH RING ABOVE
    '\u0170'   #  0xF4 -> LATIN CAPITAL LETTER U WITH DOUBLE ACUTE
    '\u0171'   #  0xF5 -> LATIN SMALL LETTER U WITH DOUBLE ACUTE
    '\u0172'   #  0xF6 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\u0173'   #  0xF7 -> LATIN SMALL LETTER U WITH OGONEK
    '\xdd'     #  0xF8 -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xfd'     #  0xF9 -> LATIN SMALL LETTER Y WITH ACUTE
    '\u0137'   #  0xFA -> LATIN SMALL LETTER K WITH CEDILLA
    '\u017b'   #  0xFB -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\u0141'   #  0xFC -> LATIN CAPITAL LETTER L WITH STROKE
    '\u017c'   #  0xFD -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u0122'   #  0xFE -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u02c7'   #  0xFF -> CARON
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
