""" Python Character Mapping Codec mac_turkish generated from 'MAPPINGS/VENDORS/APPLE/TURKISH.TXT' with gencodec.py.

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
        name='mac-turkish',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    '\x00'     #  0x00 -> CONTROL CHARACTER
    '\x01'     #  0x01 -> CONTROL CHARACTER
    '\x02'     #  0x02 -> CONTROL CHARACTER
    '\x03'     #  0x03 -> CONTROL CHARACTER
    '\x04'     #  0x04 -> CONTROL CHARACTER
    '\x05'     #  0x05 -> CONTROL CHARACTER
    '\x06'     #  0x06 -> CONTROL CHARACTER
    '\x07'     #  0x07 -> CONTROL CHARACTER
    '\x08'     #  0x08 -> CONTROL CHARACTER
    '\t'       #  0x09 -> CONTROL CHARACTER
    '\n'       #  0x0A -> CONTROL CHARACTER
    '\x0b'     #  0x0B -> CONTROL CHARACTER
    '\x0c'     #  0x0C -> CONTROL CHARACTER
    '\r'       #  0x0D -> CONTROL CHARACTER
    '\x0e'     #  0x0E -> CONTROL CHARACTER
    '\x0f'     #  0x0F -> CONTROL CHARACTER
    '\x10'     #  0x10 -> CONTROL CHARACTER
    '\x11'     #  0x11 -> CONTROL CHARACTER
    '\x12'     #  0x12 -> CONTROL CHARACTER
    '\x13'     #  0x13 -> CONTROL CHARACTER
    '\x14'     #  0x14 -> CONTROL CHARACTER
    '\x15'     #  0x15 -> CONTROL CHARACTER
    '\x16'     #  0x16 -> CONTROL CHARACTER
    '\x17'     #  0x17 -> CONTROL CHARACTER
    '\x18'     #  0x18 -> CONTROL CHARACTER
    '\x19'     #  0x19 -> CONTROL CHARACTER
    '\x1a'     #  0x1A -> CONTROL CHARACTER
    '\x1b'     #  0x1B -> CONTROL CHARACTER
    '\x1c'     #  0x1C -> CONTROL CHARACTER
    '\x1d'     #  0x1D -> CONTROL CHARACTER
    '\x1e'     #  0x1E -> CONTROL CHARACTER
    '\x1f'     #  0x1F -> CONTROL CHARACTER
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
    '\x7f'     #  0x7F -> CONTROL CHARACTER
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0x81 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe3'     #  0x8B -> LATIN SMALL LETTER A WITH TILDE
    '\xe5'     #  0x8C -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\xec'     #  0x93 -> LATIN SMALL LETTER I WITH GRAVE
    '\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf2'     #  0x98 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf5'     #  0x9B -> LATIN SMALL LETTER O WITH TILDE
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u2020'   #  0xA0 -> DAGGER
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u2122'   #  0xAA -> TRADE MARK SIGN
    '\xb4'     #  0xAB -> ACUTE ACCENT
    '\xa8'     #  0xAC -> DIAERESIS
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\xc6'     #  0xAE -> LATIN CAPITAL LETTER AE
    '\xd8'     #  0xAF -> LATIN CAPITAL LETTER O WITH STROKE
    '\u221e'   #  0xB0 -> INFINITY
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\xa5'     #  0xB4 -> YEN SIGN
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\u2202'   #  0xB6 -> PARTIAL DIFFERENTIAL
    '\u2211'   #  0xB7 -> N-ARY SUMMATION
    '\u220f'   #  0xB8 -> N-ARY PRODUCT
    '\u03c0'   #  0xB9 -> GREEK SMALL LETTER PI
    '\u222b'   #  0xBA -> INTEGRAL
    '\xaa'     #  0xBB -> FEMININE ORDINAL INDICATOR
    '\xba'     #  0xBC -> MASCULINE ORDINAL INDICATOR
    '\u03a9'   #  0xBD -> GREEK CAPITAL LETTER OMEGA
    '\xe6'     #  0xBE -> LATIN SMALL LETTER AE
    '\xf8'     #  0xBF -> LATIN SMALL LETTER O WITH STROKE
    '\xbf'     #  0xC0 -> INVERTED QUESTION MARK
    '\xa1'     #  0xC1 -> INVERTED EXCLAMATION MARK
    '\xac'     #  0xC2 -> NOT SIGN
    '\u221a'   #  0xC3 -> SQUARE ROOT
    '\u0192'   #  0xC4 -> LATIN SMALL LETTER F WITH HOOK
    '\u2248'   #  0xC5 -> ALMOST EQUAL TO
    '\u2206'   #  0xC6 -> INCREMENT
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xC8 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\xc0'     #  0xCB -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc3'     #  0xCC -> LATIN CAPITAL LETTER A WITH TILDE
    '\xd5'     #  0xCD -> LATIN CAPITAL LETTER O WITH TILDE
    '\u0152'   #  0xCE -> LATIN CAPITAL LIGATURE OE
    '\u0153'   #  0xCF -> LATIN SMALL LIGATURE OE
    '\u2013'   #  0xD0 -> EN DASH
    '\u2014'   #  0xD1 -> EM DASH
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u25ca'   #  0xD7 -> LOZENGE
    '\xff'     #  0xD8 -> LATIN SMALL LETTER Y WITH DIAERESIS
    '\u0178'   #  0xD9 -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\u011e'   #  0xDA -> LATIN CAPITAL LETTER G WITH BREVE
    '\u011f'   #  0xDB -> LATIN SMALL LETTER G WITH BREVE
    '\u0130'   #  0xDC -> LATIN CAPITAL LETTER I WITH DOT ABOVE
    '\u0131'   #  0xDD -> LATIN SMALL LETTER DOTLESS I
    '\u015e'   #  0xDE -> LATIN CAPITAL LETTER S WITH CEDILLA
    '\u015f'   #  0xDF -> LATIN SMALL LETTER S WITH CEDILLA
    '\u2021'   #  0xE0 -> DOUBLE DAGGER
    '\xb7'     #  0xE1 -> MIDDLE DOT
    '\u201a'   #  0xE2 -> SINGLE LOW-9 QUOTATION MARK
    '\u201e'   #  0xE3 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2030'   #  0xE4 -> PER MILLE SIGN
    '\xc2'     #  0xE5 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xca'     #  0xE6 -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xc1'     #  0xE7 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xcb'     #  0xE8 -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xc8'     #  0xE9 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xcd'     #  0xEA -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xEB -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xEC -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xcc'     #  0xED -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xd3'     #  0xEE -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xEF -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\uf8ff'   #  0xF0 -> Apple logo
    '\xd2'     #  0xF1 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xda'     #  0xF2 -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xF3 -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xd9'     #  0xF4 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\uf8a0'   #  0xF5 -> undefined1
    '\u02c6'   #  0xF6 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u02dc'   #  0xF7 -> SMALL TILDE
    '\xaf'     #  0xF8 -> MACRON
    '\u02d8'   #  0xF9 -> BREVE
    '\u02d9'   #  0xFA -> DOT ABOVE
    '\u02da'   #  0xFB -> RING ABOVE
    '\xb8'     #  0xFC -> CEDILLA
    '\u02dd'   #  0xFD -> DOUBLE ACUTE ACCENT
    '\u02db'   #  0xFE -> OGONEK
    '\u02c7'   #  0xFF -> CARON
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
