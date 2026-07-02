""" Python Character Mapping Codec koi8_u generated from 'python-mappings/KOI8-U.TXT' with gencodec.py.

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
        name='koi8-u',
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
    '\u2500'   #  0x80 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u2502'   #  0x81 -> BOX DRAWINGS LIGHT VERTICAL
    '\u250c'   #  0x82 -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2510'   #  0x83 -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x84 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2518'   #  0x85 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u251c'   #  0x86 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2524'   #  0x87 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u252c'   #  0x88 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u2534'   #  0x89 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u253c'   #  0x8A -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u2580'   #  0x8B -> UPPER HALF BLOCK
    '\u2584'   #  0x8C -> LOWER HALF BLOCK
    '\u2588'   #  0x8D -> FULL BLOCK
    '\u258c'   #  0x8E -> LEFT HALF BLOCK
    '\u2590'   #  0x8F -> RIGHT HALF BLOCK
    '\u2591'   #  0x90 -> LIGHT SHADE
    '\u2592'   #  0x91 -> MEDIUM SHADE
    '\u2593'   #  0x92 -> DARK SHADE
    '\u2320'   #  0x93 -> TOP HALF INTEGRAL
    '\u25a0'   #  0x94 -> BLACK SQUARE
    '\u2219'   #  0x95 -> BULLET OPERATOR
    '\u221a'   #  0x96 -> SQUARE ROOT
    '\u2248'   #  0x97 -> ALMOST EQUAL TO
    '\u2264'   #  0x98 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0x99 -> GREATER-THAN OR EQUAL TO
    '\xa0'     #  0x9A -> NO-BREAK SPACE
    '\u2321'   #  0x9B -> BOTTOM HALF INTEGRAL
    '\xb0'     #  0x9C -> DEGREE SIGN
    '\xb2'     #  0x9D -> SUPERSCRIPT TWO
    '\xb7'     #  0x9E -> MIDDLE DOT
    '\xf7'     #  0x9F -> DIVISION SIGN
    '\u2550'   #  0xA0 -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u2551'   #  0xA1 -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2552'   #  0xA2 -> BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    '\u0451'   #  0xA3 -> CYRILLIC SMALL LETTER IO
    '\u0454'   #  0xA4 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u2554'   #  0xA5 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u0456'   #  0xA6 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0457'   #  0xA7 -> CYRILLIC SMALL LETTER YI (UKRAINIAN)
    '\u2557'   #  0xA8 -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u2558'   #  0xA9 -> BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    '\u2559'   #  0xAA -> BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    '\u255a'   #  0xAB -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u255b'   #  0xAC -> BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    '\u0491'   #  0xAD -> CYRILLIC SMALL LETTER UKRAINIAN GHE WITH UPTURN
    '\u255d'   #  0xAE -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u255e'   #  0xAF -> BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    '\u255f'   #  0xB0 -> BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    '\u2560'   #  0xB1 -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2561'   #  0xB2 -> BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    '\u0401'   #  0xB3 -> CYRILLIC CAPITAL LETTER IO
    '\u0404'   #  0xB4 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u2563'   #  0xB5 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u0406'   #  0xB6 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0407'   #  0xB7 -> CYRILLIC CAPITAL LETTER YI (UKRAINIAN)
    '\u2566'   #  0xB8 -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2567'   #  0xB9 -> BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    '\u2568'   #  0xBA -> BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    '\u2569'   #  0xBB -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u256a'   #  0xBC -> BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    '\u0490'   #  0xBD -> CYRILLIC CAPITAL LETTER UKRAINIAN GHE WITH UPTURN
    '\u256c'   #  0xBE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\xa9'     #  0xBF -> COPYRIGHT SIGN
    '\u044e'   #  0xC0 -> CYRILLIC SMALL LETTER YU
    '\u0430'   #  0xC1 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xC2 -> CYRILLIC SMALL LETTER BE
    '\u0446'   #  0xC3 -> CYRILLIC SMALL LETTER TSE
    '\u0434'   #  0xC4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xC5 -> CYRILLIC SMALL LETTER IE
    '\u0444'   #  0xC6 -> CYRILLIC SMALL LETTER EF
    '\u0433'   #  0xC7 -> CYRILLIC SMALL LETTER GHE
    '\u0445'   #  0xC8 -> CYRILLIC SMALL LETTER HA
    '\u0438'   #  0xC9 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xCA -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xCB -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xCC -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xCD -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xCE -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xCF -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xD0 -> CYRILLIC SMALL LETTER PE
    '\u044f'   #  0xD1 -> CYRILLIC SMALL LETTER YA
    '\u0440'   #  0xD2 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xD3 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xD4 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xD5 -> CYRILLIC SMALL LETTER U
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0432'   #  0xD7 -> CYRILLIC SMALL LETTER VE
    '\u044c'   #  0xD8 -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044b'   #  0xD9 -> CYRILLIC SMALL LETTER YERU
    '\u0437'   #  0xDA -> CYRILLIC SMALL LETTER ZE
    '\u0448'   #  0xDB -> CYRILLIC SMALL LETTER SHA
    '\u044d'   #  0xDC -> CYRILLIC SMALL LETTER E
    '\u0449'   #  0xDD -> CYRILLIC SMALL LETTER SHCHA
    '\u0447'   #  0xDE -> CYRILLIC SMALL LETTER CHE
    '\u044a'   #  0xDF -> CYRILLIC SMALL LETTER HARD SIGN
    '\u042e'   #  0xE0 -> CYRILLIC CAPITAL LETTER YU
    '\u0410'   #  0xE1 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xE2 -> CYRILLIC CAPITAL LETTER BE
    '\u0426'   #  0xE3 -> CYRILLIC CAPITAL LETTER TSE
    '\u0414'   #  0xE4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xE5 -> CYRILLIC CAPITAL LETTER IE
    '\u0424'   #  0xE6 -> CYRILLIC CAPITAL LETTER EF
    '\u0413'   #  0xE7 -> CYRILLIC CAPITAL LETTER GHE
    '\u0425'   #  0xE8 -> CYRILLIC CAPITAL LETTER HA
    '\u0418'   #  0xE9 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xEA -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xEB -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xEC -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xED -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xEE -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xEF -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xF0 -> CYRILLIC CAPITAL LETTER PE
    '\u042f'   #  0xF1 -> CYRILLIC CAPITAL LETTER YA
    '\u0420'   #  0xF2 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xF3 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xF4 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xF5 -> CYRILLIC CAPITAL LETTER U
    '\u0416'   #  0xF6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0412'   #  0xF7 -> CYRILLIC CAPITAL LETTER VE
    '\u042c'   #  0xF8 -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042b'   #  0xF9 -> CYRILLIC CAPITAL LETTER YERU
    '\u0417'   #  0xFA -> CYRILLIC CAPITAL LETTER ZE
    '\u0428'   #  0xFB -> CYRILLIC CAPITAL LETTER SHA
    '\u042d'   #  0xFC -> CYRILLIC CAPITAL LETTER E
    '\u0429'   #  0xFD -> CYRILLIC CAPITAL LETTER SHCHA
    '\u0427'   #  0xFE -> CYRILLIC CAPITAL LETTER CHE
    '\u042a'   #  0xFF -> CYRILLIC CAPITAL LETTER HARD SIGN
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
