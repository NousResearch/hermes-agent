""" Python Character Mapping Codec generated from 'hp_roman8.txt' with gencodec.py.

    Based on data from ftp://dkuug.dk/i18n/charmaps/HP-ROMAN8 (Keld Simonsen)

    Original source: LaserJet IIP Printer User's Manual HP part no
    33471-90901, Hewlet-Packard, June 1989.

    (Used with permission)

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
        name='hp-roman8',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
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
    '\x80'     #  0x80 -> <control>
    '\x81'     #  0x81 -> <control>
    '\x82'     #  0x82 -> <control>
    '\x83'     #  0x83 -> <control>
    '\x84'     #  0x84 -> <control>
    '\x85'     #  0x85 -> <control>
    '\x86'     #  0x86 -> <control>
    '\x87'     #  0x87 -> <control>
    '\x88'     #  0x88 -> <control>
    '\x89'     #  0x89 -> <control>
    '\x8a'     #  0x8A -> <control>
    '\x8b'     #  0x8B -> <control>
    '\x8c'     #  0x8C -> <control>
    '\x8d'     #  0x8D -> <control>
    '\x8e'     #  0x8E -> <control>
    '\x8f'     #  0x8F -> <control>
    '\x90'     #  0x90 -> <control>
    '\x91'     #  0x91 -> <control>
    '\x92'     #  0x92 -> <control>
    '\x93'     #  0x93 -> <control>
    '\x94'     #  0x94 -> <control>
    '\x95'     #  0x95 -> <control>
    '\x96'     #  0x96 -> <control>
    '\x97'     #  0x97 -> <control>
    '\x98'     #  0x98 -> <control>
    '\x99'     #  0x99 -> <control>
    '\x9a'     #  0x9A -> <control>
    '\x9b'     #  0x9B -> <control>
    '\x9c'     #  0x9C -> <control>
    '\x9d'     #  0x9D -> <control>
    '\x9e'     #  0x9E -> <control>
    '\x9f'     #  0x9F -> <control>
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\xc0'     #  0xA1 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc2'     #  0xA2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc8'     #  0xA3 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xca'     #  0xA4 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xA5 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xce'     #  0xA6 -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xA7 -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xb4'     #  0xA8 -> ACUTE ACCENT
    '\u02cb'   #  0xA9 -> MODIFIER LETTER GRAVE ACCENT (MANDARIN CHINESE FOURTH TONE)
    '\u02c6'   #  0xAA -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\xa8'     #  0xAB -> DIAERESIS
    '\u02dc'   #  0xAC -> SMALL TILDE
    '\xd9'     #  0xAD -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xdb'     #  0xAE -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\u20a4'   #  0xAF -> LIRA SIGN
    '\xaf'     #  0xB0 -> MACRON
    '\xdd'     #  0xB1 -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xfd'     #  0xB2 -> LATIN SMALL LETTER Y WITH ACUTE
    '\xb0'     #  0xB3 -> DEGREE SIGN
    '\xc7'     #  0xB4 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xe7'     #  0xB5 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xd1'     #  0xB6 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xf1'     #  0xB7 -> LATIN SMALL LETTER N WITH TILDE
    '\xa1'     #  0xB8 -> INVERTED EXCLAMATION MARK
    '\xbf'     #  0xB9 -> INVERTED QUESTION MARK
    '\xa4'     #  0xBA -> CURRENCY SIGN
    '\xa3'     #  0xBB -> POUND SIGN
    '\xa5'     #  0xBC -> YEN SIGN
    '\xa7'     #  0xBD -> SECTION SIGN
    '\u0192'   #  0xBE -> LATIN SMALL LETTER F WITH HOOK
    '\xa2'     #  0xBF -> CENT SIGN
    '\xe2'     #  0xC0 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xea'     #  0xC1 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xf4'     #  0xC2 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xfb'     #  0xC3 -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xe1'     #  0xC4 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe9'     #  0xC5 -> LATIN SMALL LETTER E WITH ACUTE
    '\xf3'     #  0xC6 -> LATIN SMALL LETTER O WITH ACUTE
    '\xfa'     #  0xC7 -> LATIN SMALL LETTER U WITH ACUTE
    '\xe0'     #  0xC8 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe8'     #  0xC9 -> LATIN SMALL LETTER E WITH GRAVE
    '\xf2'     #  0xCA -> LATIN SMALL LETTER O WITH GRAVE
    '\xf9'     #  0xCB -> LATIN SMALL LETTER U WITH GRAVE
    '\xe4'     #  0xCC -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xeb'     #  0xCD -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xf6'     #  0xCE -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xfc'     #  0xCF -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xc5'     #  0xD0 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xee'     #  0xD1 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xd8'     #  0xD2 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xc6'     #  0xD3 -> LATIN CAPITAL LETTER AE
    '\xe5'     #  0xD4 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xed'     #  0xD5 -> LATIN SMALL LETTER I WITH ACUTE
    '\xf8'     #  0xD6 -> LATIN SMALL LETTER O WITH STROKE
    '\xe6'     #  0xD7 -> LATIN SMALL LETTER AE
    '\xc4'     #  0xD8 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xec'     #  0xD9 -> LATIN SMALL LETTER I WITH GRAVE
    '\xd6'     #  0xDA -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0xDB -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xc9'     #  0xDC -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xef'     #  0xDD -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xdf'     #  0xDE -> LATIN SMALL LETTER SHARP S (GERMAN)
    '\xd4'     #  0xDF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xc1'     #  0xE0 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc3'     #  0xE1 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xe3'     #  0xE2 -> LATIN SMALL LETTER A WITH TILDE
    '\xd0'     #  0xE3 -> LATIN CAPITAL LETTER ETH (ICELANDIC)
    '\xf0'     #  0xE4 -> LATIN SMALL LETTER ETH (ICELANDIC)
    '\xcd'     #  0xE5 -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xcc'     #  0xE6 -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xd3'     #  0xE7 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd2'     #  0xE8 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd5'     #  0xE9 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xf5'     #  0xEA -> LATIN SMALL LETTER O WITH TILDE
    '\u0160'   #  0xEB -> LATIN CAPITAL LETTER S WITH CARON
    '\u0161'   #  0xEC -> LATIN SMALL LETTER S WITH CARON
    '\xda'     #  0xED -> LATIN CAPITAL LETTER U WITH ACUTE
    '\u0178'   #  0xEE -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\xff'     #  0xEF -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\xde'     #  0xF0 -> LATIN CAPITAL LETTER THORN (ICELANDIC)
    '\xfe'     #  0xF1 -> LATIN SMALL LETTER THORN (ICELANDIC)
    '\xb7'     #  0xF2 -> MIDDLE DOT
    '\xb5'     #  0xF3 -> MICRO SIGN
    '\xb6'     #  0xF4 -> PILCROW SIGN
    '\xbe'     #  0xF5 -> VULGAR FRACTION THREE QUARTERS
    '\u2014'   #  0xF6 -> EM DASH
    '\xbc'     #  0xF7 -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xF8 -> VULGAR FRACTION ONE HALF
    '\xaa'     #  0xF9 -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0xFA -> MASCULINE ORDINAL INDICATOR
    '\xab'     #  0xFB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u25a0'   #  0xFC -> BLACK SQUARE
    '\xbb'     #  0xFD -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xb1'     #  0xFE -> PLUS-MINUS SIGN
    '\ufffe'
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
