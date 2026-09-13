/* eslint-disable key-spacing */

export default {
    XK_VoidSymbol:                  0xffffff, /* Void symbol */

    XK_BackSpace:                   0xff08, /* Back space, back char */
    XK_Tab:                         0xff09,
    XK_Linefeed:                    0xff0a, /* Linefeed, LF */
    XK_Clear:                       0xff0b,
    XK_Return:                      0xff0d, /* Return, enter */
    XK_Pause:                       0xff13, /* Pause, hold */
    XK_Scroll_Lock:                 0xff14,
    XK_Sys_Req:                     0xff15,
    XK_Escape:                      0xff1b,
    XK_Delete:                      0xffff, /* Delete, rubout */

    /* International & multi-key character composition */

    XK_Multi_key:                   0xff20, /* Multi-key character compose */
    XK_Codeinput:                   0xff37,
    XK_SingleCandidate:             0xff3c,
    XK_MultipleCandidate:           0xff3d,
    XK_PreviousCandidate:           0xff3e,

    /* Japanese keyboard support */

    XK_Kanji:                       0xff21, /* Kanji, Kanji convert */
    XK_Muhenkan:                    0xff22, /* Cancel Conversion */
    XK_Henkan_Mode:                 0xff23, /* Start/Stop Conversion */
    XK_Henkan:                      0xff23, /* Alias for Henkan_Mode */
    XK_Romaji:                      0xff24, /* to Romaji */
    XK_Hiragana:                    0xff25, /* to Hiragana */
    XK_Katakana:                    0xff26, /* to Katakana */
    XK_Hiragana_Katakana:           0xff27, /* Hiragana/Katakana toggle */
    XK_Zenkaku:                     0xff28, /* to Zenkaku */
    XK_Hankaku:                     0xff29, /* to Hankaku */
    XK_Zenkaku_Hankaku:             0xff2a, /* Zenkaku/Hankaku toggle */
    XK_Touroku:                     0xff2b, /* Add to Dictionary */
    XK_Massyo:                      0xff2c, /* Delete from Dictionary */
    XK_Kana_Lock:                   0xff2d, /* Kana Lock */
    XK_Kana_Shift:                  0xff2e, /* Kana Shift */
    XK_Eisu_Shift:                  0xff2f, /* Alphanumeric Shift */
    XK_Eisu_toggle:                 0xff30, /* Alphanumeric toggle */
    XK_Kanji_Bangou:                0xff37, /* Codeinput */
    XK_Zen_Koho:                    0xff3d, /* Multiple/All Candidate(s) */
    XK_Mae_Koho:                    0xff3e, /* Previous Candidate */

    /* Cursor control & motion */

    XK_Home:                        0xff50,
    XK_Left:                        0xff51, /* Move left, left arrow */
    XK_Up:                          0xff52, /* Move up, up arrow */
    XK_Right:                       0xff53, /* Move right, right arrow */
    XK_Down:                        0xff54, /* Move down, down arrow */
    XK_Prior:                       0xff55, /* Prior, previous */
    XK_Page_Up:                     0xff55,
    XK_Next:                        0xff56, /* Next */
    XK_Page_Down:                   0xff56,
    XK_End:                         0xff57, /* EOL */
    XK_Begin:                       0xff58, /* BOL */


    /* Misc functions */

    XK_Select:                      0xff60, /* Select, mark */
    XK_Print:                       0xff61,
    XK_Execute:                     0xff62, /* Execute, run, do */
    XK_Insert:                      0xff63, /* Insert, insert here */
    XK_Undo:                        0xff65,
    XK_Redo:                        0xff66, /* Redo, again */
    XK_Menu:                        0xff67,
    XK_Find:                        0xff68, /* Find, search */
    XK_Cancel:                      0xff69, /* Cancel, stop, abort, exit */
    XK_Help:                        0xff6a, /* Help */
    XK_Break:                       0xff6b,
    XK_Mode_switch:                 0xff7e, /* Character set switch */
    XK_script_switch:               0xff7e, /* Alias for mode_switch */
    XK_Num_Lock:                    0xff7f,

    /* Keypad functions, keypad numbers cleverly chosen to map to ASCII */

    XK_KP_Space:                    0xff80, /* Space */
    XK_KP_Tab:                      0xff89,
    XK_KP_Enter:                    0xff8d, /* Enter */
    XK_KP_F1:                       0xff91, /* PF1, KP_A, ... */
    XK_KP_F2:                       0xff92,
    XK_KP_F3:                       0xff93,
    XK_KP_F4:                       0xff94,
    XK_KP_Home:                     0xff95,
    XK_KP_Left:                     0xff96,
    XK_KP_Up:                       0xff97,
    XK_KP_Right:                    0xff98,
    XK_KP_Down:                     0xff99,
    XK_KP_Prior:                    0xff9a,
    XK_KP_Page_Up:                  0xff9a,
    XK_KP_Next:                     0xff9b,
    XK_KP_Page_Down:                0xff9b,
    XK_KP_End:                      0xff9c,
    XK_KP_Begin:                    0xff9d,
    XK_KP_Insert:                   0xff9e,
    XK_KP_Delete:                   0xff9f,
    XK_KP_Equal:                    0xffbd, /* Equals */
    XK_KP_Multiply:                 0xffaa,
    XK_KP_Add:                      0xffab,
    XK_KP_Separator:                0xffac, /* Separator, often comma */
    XK_KP_Subtract:                 0xffad,
    XK_KP_Decimal:                  0xffae,
    XK_KP_Divide:                   0xffaf,

    XK_KP_0:                        0xffb0,
    XK_KP_1:                        0xffb1,
    XK_KP_2:                        0xffb2,
    XK_KP_3:                        0xffb3,
    XK_KP_4:                        0xffb4,
    XK_KP_5:                        0xffb5,
    XK_KP_6:                        0xffb6,
    XK_KP_7:                        0xffb7,
    XK_KP_8:                        0xffb8,
    XK_KP_9:                        0xffb9,

    /*
     * Auxiliary functions; note the duplicate definitions for left and right
     * function keys;  Sun keyboards and a few other manufacturers have such
     * function key groups on the left and/or right sides of the keyboard.
     * We've not found a keyboard with more than 35 function keys total.
     */

    XK_F1:                          0xffbe,
    XK_F2:                          0xffbf,
    XK_F3:                          0xffc0,
    XK_F4:                          0xffc1,
    XK_F5:                          0xffc2,
    XK_F6:                          0xffc3,
    XK_F7:                          0xffc4,
    XK_F8:                          0xffc5,
    XK_F9:                          0xffc6,
    XK_F10:                         0xffc7,
    XK_F11:                         0xffc8,
    XK_L1:                          0xffc8,
    XK_F12:                         0xffc9,
    XK_L2:                          0xffc9,
    XK_F13:                         0xffca,
    XK_L3:                          0xffca,
    XK_F14:                         0xffcb,
    XK_L4:                          0xffcb,
    XK_F15:                         0xffcc,
    XK_L5:                          0xffcc,
    XK_F16:                         0xffcd,
    XK_L6:                          0xffcd,
    XK_F17:                         0xffce,
    XK_L7:                          0xffce,
    XK_F18:                         0xffcf,
    XK_L8:                          0xffcf,
    XK_F19:                         0xffd0,
    XK_L9:                          0xffd0,
    XK_F20:                         0xffd1,
    XK_L10:                         0xffd1,
    XK_F21:                         0xffd2,
    XK_R1:                          0xffd2,
    XK_F22:                         0xffd3,
    XK_R2:                          0xffd3,
    XK_F23:                         0xffd4,
    XK_R3:                          0xffd4,
    XK_F24:                         0xffd5,
    XK_R4:                          0xffd5,
    XK_F25:                         0xffd6,
    XK_R5:                          0xffd6,
    XK_F26:                         0xffd7,
    XK_R6:                          0xffd7,
    XK_F27:                         0xffd8,
    XK_R7:                          0xffd8,
    XK_F28:                         0xffd9,
    XK_R8:                          0xffd9,
    XK_F29:                         0xffda,
    XK_R9:                          0xffda,
    XK_F30:                         0xffdb,
    XK_R10:                         0xffdb,
    XK_F31:                         0xffdc,
    XK_R11:                         0xffdc,
    XK_F32:                         0xffdd,
    XK_R12:                         0xffdd,
    XK_F33:                         0xffde,
    XK_R13:                         0xffde,
    XK_F34:                         0xffdf,
    XK_R14:                         0xffdf,
    XK_F35:                         0xffe0,
    XK_R15:                         0xffe0,

    /* Modifiers */

    XK_Shift_L:                     0xffe1, /* Left shift */
    XK_Shift_R:                     0xffe2, /* Right shift */
    XK_Control_L:                   0xffe3, /* Left control */
    XK_Control_R:                   0xffe4, /* Right control */
    XK_Caps_Lock:                   0xffe5, /* Caps lock */
    XK_Shift_Lock:                  0xffe6, /* Shift lock */

    XK_Meta_L:                      0xffe7, /* Left meta */
    XK_Meta_R:                      0xffe8, /* Right meta */
    XK_Alt_L:                       0xffe9, /* Left alt */
    XK_Alt_R:                       0xffea, /* Right alt */
    XK_Super_L:                     0xffeb, /* Left super */
    XK_Super_R:                     0xffec, /* Right super */
    XK_Hyper_L:                     0xffed, /* Left hyper */
    XK_Hyper_R:                     0xffee, /* Right hyper */

    /*
     * Keyboard (XKB) Extension function and modifier keys
     * (from Appendix C of "The X Keyboard Extension: Protocol Specification")
     * Byte 3 = 0xfe
     */

    XK_ISO_Level3_Shift:            0xfe03, /* AltGr */
    XK_ISO_Next_Group:              0xfe08,
    XK_ISO_Prev_Group:              0xfe0a,
    XK_ISO_First_Group:             0xfe0c,
    XK_ISO_Last_Group:              0xfe0e,

    /*
     * Latin 1
     * (ISO/IEC 8859-1: Unicode U+0020..U+00FF)
     * Byte 3: 0
     */

    XK_space:                       0x0020, /* U+0020 SPACE */
    XK_exclam:                      0x0021, /* U+0021 EXCLAMATION MARK */
    XK_quotedbl:                    0x0022, /* U+0022 QUOTATION MARK */
    XK_numbersign:                  0x0023, /* U+0023 NUMBER SIGN */
    XK_dollar:                      0x0024, /* U+0024 DOLLAR SIGN */
    XK_percent:                     0x0025, /* U+0025 PERCENT SIGN */
    XK_ampersand:                   0x0026, /* U+0026 AMPERSAND */
    XK_apostrophe:                  0x0027, /* U+0027 APOSTROPHE */
    XK_quoteright:                  0x0027, /* deprecated */
    XK_parenleft:                   0x0028, /* U+0028 LEFT PARENTHESIS */
    XK_parenright:                  0x0029, /* U+0029 RIGHT PARENTHESIS */
    XK_asterisk:                    0x002a, /* U+002A ASTERISK */
    XK_plus:                        0x002b, /* U+002B PLUS SIGN */
    XK_comma:                       0x002c, /* U+002C COMMA */
    XK_minus:                       0x002d, /* U+002D HYPHEN-MINUS */
    XK_period:                      0x002e, /* U+002E FULL STOP */
    XK_slash:                       0x002f, /* U+002F SOLIDUS */
    XK_0:                           0x0030, /* U+0030 DIGIT ZERO */
    XK_1:                           0x0031, /* U+0031 DIGIT ONE */
    XK_2:                           0x0032, /* U+0032 DIGIT TWO */
    XK_3:                           0x0033, /* U+0033 DIGIT THREE */
    XK_4:                           0x0034, /* U+0034 DIGIT FOUR */
    XK_5:                           0x0035, /* U+0035 DIGIT FIVE */
    XK_6:                           0x0036, /* U+0036 DIGIT SIX */
    XK_7:                           0x0037, /* U+0037 DIGIT SEVEN */
    XK_8:                           0x0038, /* U+0038 DIGIT EIGHT */
    XK_9:                           0x0039, /* U+0039 DIGIT NINE */
    XK_colon:                       0x003a, /* U+003A COLON */
    XK_semicolon:                   0x003b, /* U+003B SEMICOLON */
    XK_less:                        0x003c, /* U+003C LESS-THAN SIGN */
    XK_equal:                       0x003d, /* U+003D EQUALS SIGN */
    XK_greater:                     0x003e, /* U+003E GREATER-THAN SIGN */
    XK_question:                    0x003f, /* U+003F QUESTION MARK */
    XK_at:                          0x0040, /* U+0040 COMMERCIAL AT */
    XK_A:                           0x0041, /* U+0041 LATIN CAPITAL LETTER A */
    XK_B:                           0x0042, /* U+0042 LATIN CAPITAL LETTER B */
    XK_C:                           0x0043, /* U+0043 LATIN CAPITAL LETTER C */
    XK_D:                           0x0044, /* U+0044 LATIN CAPITAL LETTER D */
    XK_E:                           0x0045, /* U+0045 LATIN CAPITAL LETTER E */
    XK_F:                           0x0046, /* U+0046 LATIN CAPITAL LETTER F */
    XK_G:                           0x0047, /* U+0047 LATIN CAPITAL LETTER G */
    XK_H:                           0x0048, /* U+0048 LATIN CAPITAL LETTER H */
    XK_I:                           0x0049, /* U+0049 LATIN CAPITAL LETTER I */
    XK_J:                           0x004a, /* U+004A LATIN CAPITAL LETTER J */
    XK_K:                           0x004b, /* U+004B LATIN CAPITAL LETTER K */
    XK_L:                           0x004c, /* U+004C LATIN CAPITAL LETTER L */
    XK_M:                           0x004d, /* U+004D LATIN CAPITAL LETTER M */
    XK_N:                           0x004e, /* U+004E LATIN CAPITAL LETTER N */
    XK_O:                           0x004f, /* U+004F LATIN CAPITAL LETTER O */
    XK_P:                           0x0050, /* U+0050 LATIN CAPITAL LETTER P */
    XK_Q:                           0x0051, /* U+0051 LATIN CAPITAL LETTER Q */
    XK_R:                           0x0052, /* U+0052 LATIN CAPITAL LETTER R */
    XK_S:                           0x0053, /* U+0053 LATIN CAPITAL LETTER S */
    XK_T:                           0x0054, /* U+0054 LATIN CAPITAL LETTER T */
    XK_U:                           0x0055, /* U+0055 LATIN CAPITAL LETTER U */
    XK_V:                           0x0056, /* U+0056 LATIN CAPITAL LETTER V */
    XK_W:                           0x0057, /* U+0057 LATIN CAPITAL LETTER W */
    XK_X:                           0x0058, /* U+0058 LATIN CAPITAL LETTER X */
    XK_Y:                           0x0059, /* U+0059 LATIN CAPITAL LETTER Y */
    XK_Z:                           0x005a, /* U+005A LATIN CAPITAL LETTER Z */
    XK_bracketleft:                 0x005b, /* U+005B LEFT SQUARE BRACKET */
    XK_backslash:                   0x005c, /* U+005C REVERSE SOLIDUS */
    XK_bracketright:                0x005d, /* U+005D RIGHT SQUARE BRACKET */
    XK_asciicircum:                 0x005e, /* U+005E CIRCUMFLEX ACCENT */
    XK_underscore:                  0x005f, /* U+005F LOW LINE */
    XK_grave:                       0x0060, /* U+0060 GRAVE ACCENT */
    XK_quoteleft:                   0x0060, /* deprecated */
    XK_a:                           0x0061, /* U+0061 LATIN SMALL LETTER A */
    XK_b:                           0x0062, /* U+0062 LATIN SMALL LETTER B */
    XK_c:                           0x0063, /* U+0063 LATIN SMALL LETTER C */
    XK_d:                           0x0064, /* U+0064 LATIN SMALL LETTER D */
    XK_e:                           0x0065, /* U+0065 LATIN SMALL LETTER E */
    XK_f:                           0x0066, /* U+0066 LATIN SMALL LETTER F */
    XK_g:                           0x0067, /* U+0067 LATIN SMALL LETTER G */
    XK_h:                           0x0068, /* U+0068 LATIN SMALL LETTER H */
    XK_i:                           0x0069, /* U+0069 LATIN SMALL LETTER I */
    XK_j:                           0x006a, /* U+006A LATIN SMALL LETTER J */
    XK_k:                           0x006b, /* U+006B LATIN SMALL LETTER K */
    XK_l:                           0x006c, /* U+006C LATIN SMALL LETTER L */
    XK_m:                           0x006d, /* U+006D LATIN SMALL LETTER M */
    XK_n:                           0x006e, /* U+006E LATIN SMALL LETTER N */
    XK_o:                           0x006f, /* U+006F LATIN SMALL LETTER O */
    XK_p:                           0x0070, /* U+0070 LATIN SMALL LETTER P */
    XK_q:                           0x0071, /* U+0071 LATIN SMALL LETTER Q */
    XK_r:                           0x0072, /* U+0072 LATIN SMALL LETTER R */
    XK_s:                           0x0073, /* U+0073 LATIN SMALL LETTER S */
    XK_t:                           0x0074, /* U+0074 LATIN SMALL LETTER T */
    XK_u:                           0x0075, /* U+0075 LATIN SMALL LETTER U */
    XK_v:                           0x0076, /* U+0076 LATIN SMALL LETTER V */
    XK_w:                           0x0077, /* U+0077 LATIN SMALL LETTER W */
    XK_x:                           0x0078, /* U+0078 LATIN SMALL LETTER X */
    XK_y:                           0x0079, /* U+0079 LATIN SMALL LETTER Y */
    XK_z:                           0x007a, /* U+007A LATIN SMALL LETTER Z */
    XK_braceleft:                   0x007b, /* U+007B LEFT CURLY BRACKET */
    XK_bar:                         0x007c, /* U+007C VERTICAL LINE */
    XK_braceright:                  0x007d, /* U+007D RIGHT CURLY BRACKET */
    XK_asciitilde:                  0x007e, /* U+007E TILDE */

    XK_nobreakspace:                0x00a0, /* U+00A0 NO-BREAK SPACE */
    XK_exclamdown:                  0x00a1, /* U+00A1 INVERTED EXCLAMATION MARK */
    XK_cent:                        0x00a2, /* U+00A2 CENT SIGN */
    XK_sterling:                    0x00a3, /* U+00A3 POUND SIGN */
    XK_currency:                    0x00a4, /* U+00A4 CURRENCY SIGN */
    XK_yen:                         0x00a5, /* U+00A5 YEN SIGN */
    XK_brokenbar:                   0x00a6, /* U+00A6 BROKEN BAR */
    XK_section:                     0x00a7, /* U+00A7 SECTION SIGN */
    XK_diaeresis:                   0x00a8, /* U+00A8 DIAERESIS */
    XK_copyright:                   0x00a9, /* U+00A9 COPYRIGHT SIGN */
    XK_ordfeminine:                 0x00aa, /* U+00AA FEMININE ORDINAL INDICATOR */
    XK_guillemotleft:               0x00ab, /* U+00AB LEFT-POINTING DOUBLE ANGLE QUOTATION MARK */
    XK_notsign:                     0x00ac, /* U+00AC NOT SIGN */
    XK_hyphen:                      0x00ad, /* U+00AD SOFT HYPHEN */
    XK_registered:                  0x00ae, /* U+00AE REGISTERED SIGN */
    XK_macron:                      0x00af, /* U+00AF MACRON */
    XK_degree:                      0x00b0, /* U+00B0 DEGREE SIGN */
    XK_plusminus:                   0x00b1, /* U+00B1 PLUS-MINUS SIGN */
    XK_twosuperior:                 0x00b2, /* U+00B2 SUPERSCRIPT TWO */
    XK_threesuperior:               0x00b3, /* U+00B3 SUPERSCRIPT THREE */
    XK_acute:                       0x00b4, /* U+00B4 ACUTE ACCENT */
    XK_mu:                          0x00b5, /* U+00B5 MICRO SIGN */
    XK_paragraph:                   0x00b6, /* U+00B6 PILCROW SIGN */
    XK_periodcentered:              0x00b7, /* U+00B7 MIDDLE DOT */
    XK_cedilla:                     0x00b8, /* U+00B8 CEDILLA */
    XK_onesuperior:                 0x00b9, /* U+00B9 SUPERSCRIPT ONE */
    XK_masculine:                   0x00ba, /* U+00BA MASCULINE ORDINAL INDICATOR */
    XK_guillemotright:              0x00bb, /* U+00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK */
    XK_onequarter:                  0x00bc, /* U+00BC VULGAR FRACTION ONE QUARTER */
    XK_onehalf:                     0x00bd, /* U+00BD VULGAR FRACTION ONE HALF */
    XK_threequarters:               0x00be, /* U+00BE VULGAR FRACTION THREE QUARTERS */
    XK_questiondown:                0x00bf, /* U+00BF INVERTED QUESTION MARK */
    XK_Agrave:                      0x00c0, /* U+00C0 LATIN CAPITAL LETTER A WITH GRAVE */
    XK_Aacute:                      0x00c1, /* U+00C1 LATIN CAPITAL LETTER A WITH ACUTE */
    XK_Acircumflex:                 0x00c2, /* U+00C2 LATIN CAPITAL LETTER A WITH CIRCUMFLEX */
    XK_Atilde:                      0x00c3, /* U+00C3 LATIN CAPITAL LETTER A WITH TILDE */
    XK_Adiaeresis:                  0x00c4, /* U+00C4 LATIN CAPITAL LETTER A WITH DIAERESIS */
    XK_Aring:                       0x00c5, /* U+00C5 LATIN CAPITAL LETTER A WITH RING ABOVE */
    XK_AE:                          0x00c6, /* U+00C6 LATIN CAPITAL LETTER AE */
    XK_Ccedilla:                    0x00c7, /* U+00C7 LATIN CAPITAL LETTER C WITH CEDILLA */
    XK_Egrave:                      0x00c8, /* U+00C8 LATIN CAPITAL LETTER E WITH GRAVE */
    XK_Eacute:                      0x00c9, /* U+00C9 LATIN CAPITAL LETTER E WITH ACUTE */
    XK_Ecircumflex:                 0x00ca, /* U+00CA LATIN CAPITAL LETTER E WITH CIRCUMFLEX */
    XK_Ediaeresis:                  0x00cb, /* U+00CB LATIN CAPITAL LETTER E WITH DIAERESIS */
    XK_Igrave:                      0x00cc, /* U+00CC LATIN CAPITAL LETTER I WITH GRAVE */
    XK_Iacute:                      0x00cd, /* U+00CD LATIN CAPITAL LETTER I WITH ACUTE */
    XK_Icircumflex:                 0x00ce, /* U+00CE LATIN CAPITAL LETTER I WITH CIRCUMFLEX */
    XK_Idiaeresis:                  0x00cf, /* U+00CF LATIN CAPITAL LETTER I WITH DIAERESIS */
    XK_ETH:                         0x00d0, /* U+00D0 LATIN CAPITAL LETTER ETH */
    XK_Eth:                         0x00d0, /* deprecated */
    XK_Ntilde:                      0x00d1, /* U+00D1 LATIN CAPITAL LETTER N WITH TILDE */
    XK_Ograve:                      0x00d2, /* U+00D2 LATIN CAPITAL LETTER O WITH GRAVE */
    XK_Oacute:                      0x00d3, /* U+00D3 LATIN CAPITAL LETTER O WITH ACUTE */
    XK_Ocircumflex:                 0x00d4, /* U+00D4 LATIN CAPITAL LETTER O WITH CIRCUMFLEX */
    XK_Otilde:                      0x00d5, /* U+00D5 LATIN CAPITAL LETTER O WITH TILDE */
    XK_Odiaeresis:                  0x00d6, /* U+00D6 LATIN CAPITAL LETTER O WITH DIAERESIS */
    XK_multiply:                    0x00d7, /* U+00D7 MULTIPLICATION SIGN */
    XK_Oslash:                      0x00d8, /* U+00D8 LATIN CAPITAL LETTER O WITH STROKE */
    XK_Ooblique:                    0x00d8, /* U+00D8 LATIN CAPITAL LETTER O WITH STROKE */
    XK_Ugrave:                      0x00d9, /* U+00D9 LATIN CAPITAL LETTER U WITH GRAVE */
    XK_Uacute:                      0x00da, /* U+00DA LATIN CAPITAL LETTER U WITH ACUTE */
    XK_Ucircumflex:                 0x00db, /* U+00DB LATIN CAPITAL LETTER U WITH CIRCUMFLEX */
    XK_Udiaeresis:                  0x00dc, /* U+00DC LATIN CAPITAL LETTER U WITH DIAERESIS */
    XK_Yacute:                      0x00dd, /* U+00DD LATIN CAPITAL LETTER Y WITH ACUTE */
    XK_THORN:                       0x00de, /* U+00DE LATIN CAPITAL LETTER THORN */
    XK_Thorn:                       0x00de, /* deprecated */
    XK_ssharp:                      0x00df, /* U+00DF LATIN SMALL LETTER SHARP S */
    XK_agrave:                      0x00e0, /* U+00E0 LATIN SMALL LETTER A WITH GRAVE */
    XK_aacute:                      0x00e1, /* U+00E1 LATIN SMALL LETTER A WITH ACUTE */
    XK_acircumflex:                 0x00e2, /* U+00E2 LATIN SMALL LETTER A WITH CIRCUMFLEX */
    XK_atilde:                      0x00e3, /* U+00E3 LATIN SMALL LETTER A WITH TILDE */
    XK_adiaeresis:                  0x00e4, /* U+00E4 LATIN SMALL LETTER A WITH DIAERESIS */
    XK_aring:                       0x00e5, /* U+00E5 LATIN SMALL LETTER A WITH RING ABOVE */
    XK_ae:                          0x00e6, /* U+00E6 LATIN SMALL LETTER AE */
    XK_ccedilla:                    0x00e7, /* U+00E7 LATIN SMALL LETTER C WITH CEDILLA */
    XK_egrave:                      0x00e8, /* U+00E8 LATIN SMALL LETTER E WITH GRAVE */
    XK_eacute:                      0x00e9, /* U+00E9 LATIN SMALL LETTER E WITH ACUTE */
    XK_ecircumflex:                 0x00ea, /* U+00EA LATIN SMALL LETTER E WITH CIRCUMFLEX */
    XK_ediaeresis:                  0x00eb, /* U+00EB LATIN SMALL LETTER E WITH DIAERESIS */
    XK_igrave:                      0x00ec, /* U+00EC LATIN SMALL LETTER I WITH GRAVE */
    XK_iacute:                      0x00ed, /* U+00ED LATIN SMALL LETTER I WITH ACUTE */
    XK_icircumflex:                 0x00ee, /* U+00EE LATIN SMALL LETTER I WITH CIRCUMFLEX */
    XK_idiaeresis:                  0x00ef, /* U+00EF LATIN SMALL LETTER I WITH DIAERESIS */
    XK_eth:                         0x00f0, /* U+00F0 LATIN SMALL LETTER ETH */
    XK_ntilde:                      0x00f1, /* U+00F1 LATIN SMALL LETTER N WITH TILDE */
    XK_ograve:                      0x00f2, /* U+00F2 LATIN SMALL LETTER O WITH GRAVE */
    XK_oacute:                      0x00f3, /* U+00F3 LATIN SMALL LETTER O WITH ACUTE */
    XK_ocircumflex:                 0x00f4, /* U+00F4 LATIN SMALL LETTER O WITH CIRCUMFLEX */
    XK_otilde:                      0x00f5, /* U+00F5 LATIN SMALL LETTER O WITH TILDE */
    XK_odiaeresis:                  0x00f6, /* U+00F6 LATIN SMALL LETTER O WITH DIAERESIS */
    XK_division:                    0x00f7, /* U+00F7 DIVISION SIGN */
    XK_oslash:                      0x00f8, /* U+00F8 LATIN SMALL LETTER O WITH STROKE */
    XK_ooblique:                    0x00f8, /* U+00F8 LATIN SMALL LETTER O WITH STROKE */
    XK_ugrave:                      0x00f9, /* U+00F9 LATIN SMALL LETTER U WITH GRAVE */
    XK_uacute:                      0x00fa, /* U+00FA LATIN SMALL LETTER U WITH ACUTE */
    XK_ucircumflex:                 0x00fb, /* U+00FB LATIN SMALL LETTER U WITH CIRCUMFLEX */
    XK_udiaeresis:                  0x00fc, /* U+00FC LATIN SMALL LETTER U WITH DIAERESIS */
    XK_yacute:                      0x00fd, /* U+00FD LATIN SMALL LETTER Y WITH ACUTE */
    XK_thorn:                       0x00fe, /* U+00FE LATIN SMALL LETTER THORN */
    XK_ydiaeresis:                  0x00ff, /* U+00FF LATIN SMALL LETTER Y WITH DIAERESIS */

    /*
     * Korean
     * Byte 3 = 0x0e
     */

    XK_Hangul:                      0xff31, /* Hangul start/stop(toggle) */
    XK_Hangul_Hanja:                0xff34, /* Start Hangul->Hanja Conversion */
    XK_Hangul_Jeonja:               0xff38, /* Jeonja mode */

    /*
     * XFree86 vendor specific keysyms.
     *
     * The XFree86 keysym range is 0x10080001 - 0x1008FFFF.
     */

    XF86XK_ModeLock:                0x1008FF01,
    XF86XK_MonBrightnessUp:         0x1008FF02,
    XF86XK_MonBrightnessDown:       0x1008FF03,
    XF86XK_KbdLightOnOff:           0x1008FF04,
    XF86XK_KbdBrightnessUp:         0x1008FF05,
    XF86XK_KbdBrightnessDown:       0x1008FF06,
    XF86XK_Standby:                 0x1008FF10,
    XF86XK_AudioLowerVolume:        0x1008FF11,
    XF86XK_AudioMute:               0x1008FF12,
    XF86XK_AudioRaiseVolume:        0x1008FF13,
    XF86XK_AudioPlay:               0x1008FF14,
    XF86XK_AudioStop:               0x1008FF15,
    XF86XK_AudioPrev:               0x1008FF16,
    XF86XK_AudioNext:               0x1008FF17,
    XF86XK_HomePage:                0x1008FF18,
    XF86XK_Mail:                    0x1008FF19,
    XF86XK_Start:                   0x1008FF1A,
    XF86XK_Search:                  0x1008FF1B,
    XF86XK_AudioRecord:             0x1008FF1C,
    XF86XK_Calculator:              0x1008FF1D,
    XF86XK_Memo:                    0x1008FF1E,
    XF86XK_ToDoList:                0x1008FF1F,
    XF86XK_Calendar:                0x1008FF20,
    XF86XK_PowerDown:               0x1008FF21,
    XF86XK_ContrastAdjust:          0x1008FF22,
    XF86XK_RockerUp:                0x1008FF23,
    XF86XK_RockerDown:              0x1008FF24,
    XF86XK_RockerEnter:             0x1008FF25,
    XF86XK_Back:                    0x1008FF26,
    XF86XK_Forward:                 0x1008FF27,
    XF86XK_Stop:                    0x1008FF28,
    XF86XK_Refresh:                 0x1008FF29,
    XF86XK_PowerOff:                0x1008FF2A,
    XF86XK_WakeUp:                  0x1008FF2B,
    XF86XK_Eject:                   0x1008FF2C,
    XF86XK_ScreenSaver:             0x1008FF2D,
    XF86XK_WWW:                     0x1008FF2E,
    XF86XK_Sleep:                   0x1008FF2F,
    XF86XK_Favorites:               0x1008FF30,
    XF86XK_AudioPause:              0x1008FF31,
    XF86XK_AudioMedia:              0x1008FF32,
    XF86XK_MyComputer:              0x1008FF33,
    XF86XK_VendorHome:              0x1008FF34,
    XF86XK_LightBulb:               0x1008FF35,
    XF86XK_Shop:                    0x1008FF36,
    XF86XK_History:                 0x1008FF37,
    XF86XK_OpenURL:                 0x1008FF38,
    XF86XK_AddFavorite:             0x1008FF39,
    XF86XK_HotLinks:                0x1008FF3A,
    XF86XK_BrightnessAdjust:        0x1008FF3B,
    XF86XK_Finance:                 0x1008FF3C,
    XF86XK_Community:               0x1008FF3D,
    XF86XK_AudioRewind:             0x1008FF3E,
    XF86XK_BackForward:             0x1008FF3F,
    XF86XK_Launch0:                 0x1008FF40,
    XF86XK_Launch1:                 0x1008FF41,
    XF86XK_Launch2:                 0x1008FF42,
    XF86XK_Launch3:                 0x1008FF43,
    XF86XK_Launch4:                 0x1008FF44,
    XF86XK_Launch5:                 0x1008FF45,
    XF86XK_Launch6:                 0x1008FF46,
    XF86XK_Launch7:                 0x1008FF47,
    XF86XK_Launch8:                 0x1008FF48,
    XF86XK_Launch9:                 0x1008FF49,
    XF86XK_LaunchA:                 0x1008FF4A,
    XF86XK_LaunchB:                 0x1008FF4B,
    XF86XK_LaunchC:                 0x1008FF4C,
    XF86XK_LaunchD:                 0x1008FF4D,
    XF86XK_LaunchE:                 0x1008FF4E,
    XF86XK_LaunchF:                 0x1008FF4F,
    XF86XK_ApplicationLeft:         0x1008FF50,
    XF86XK_ApplicationRight:        0x1008FF51,
    XF86XK_Book:                    0x1008FF52,
    XF86XK_CD:                      0x1008FF53,
    XF86XK_Calculater:              0x1008FF54,
    XF86XK_Clear:                   0x1008FF55,
    XF86XK_Close:                   0x1008FF56,
    XF86XK_Copy:                    0x1008FF57,
    XF86XK_Cut:                     0x1008FF58,
    XF86XK_Display:                 0x1008FF59,
    XF86XK_DOS:                     0x1008FF5A,
    XF86XK_Documents:               0x1008FF5B,
    XF86XK_Excel:                   0x1008FF5C,
    XF86XK_Explorer:                0x1008FF5D,
    XF86XK_Game:                    0x1008FF5E,
    XF86XK_Go:                      0x1008FF5F,
    XF86XK_iTouch:                  0x1008FF60,
    XF86XK_LogOff:                  0x1008FF61,
    XF86XK_Market:                  0x1008FF62,
    XF86XK_Meeting:                 0x1008FF63,
    XF86XK_MenuKB:                  0x1008FF65,
    XF86XK_MenuPB:                  0x1008FF66,
    XF86XK_MySites:                 0x1008FF67,
    XF86XK_New:                     0x1008FF68,
    XF86XK_News:                    0x1008FF69,
    XF86XK_OfficeHome:              0x1008FF6A,
    XF86XK_Open:                    0x1008FF6B,
    XF86XK_Option:                  0x1008FF6C,
    XF86XK_Paste:                   0x1008FF6D,
    XF86XK_Phone:                   0x1008FF6E,
    XF86XK_Q:                       0x1008FF70,
    XF86XK_Reply:                   0x1008FF72,
    XF86XK_Reload:                  0x1008FF73,
    XF86XK_RotateWindows:           0x1008FF74,
    XF86XK_RotationPB:              0x1008FF75,
    XF86XK_RotationKB:              0x1008FF76,
    XF86XK_Save:                    0x1008FF77,
    XF86XK_ScrollUp:                0x1008FF78,
    XF86XK_ScrollDown:              0x1008FF79,
    XF86XK_ScrollClick:             0x1008FF7A,
    XF86XK_Send:                    0x1008FF7B,
    XF86XK_Spell:                   0x1008FF7C,
    XF86XK_SplitScreen:             0x1008FF7D,
    XF86XK_Support:                 0x1008FF7E,
    XF86XK_TaskPane:                0x1008FF7F,
    XF86XK_Terminal:                0x1008FF80,
    XF86XK_Tools:                   0x1008FF81,
    XF86XK_Travel:                  0x1008FF82,
    XF86XK_UserPB:                  0x1008FF84,
    XF86XK_User1KB:                 0x1008FF85,
    XF86XK_User2KB:                 0x1008FF86,
    XF86XK_Video:                   0x1008FF87,
    XF86XK_WheelButton:             0x1008FF88,
    XF86XK_Word:                    0x1008FF89,
    XF86XK_Xfer:                    0x1008FF8A,
    XF86XK_ZoomIn:                  0x1008FF8B,
    XF86XK_ZoomOut:                 0x1008FF8C,
    XF86XK_Away:                    0x1008FF8D,
    XF86XK_Messenger:               0x1008FF8E,
    XF86XK_WebCam:                  0x1008FF8F,
    XF86XK_MailForward:             0x1008FF90,
    XF86XK_Pictures:                0x1008FF91,
    XF86XK_Music:                   0x1008FF92,
    XF86XK_Battery:                 0x1008FF93,
    XF86XK_Bluetooth:               0x1008FF94,
    XF86XK_WLAN:                    0x1008FF95,
    XF86XK_UWB:                     0x1008FF96,
    XF86XK_AudioForward:            0x1008FF97,
    XF86XK_AudioRepeat:             0x1008FF98,
    XF86XK_AudioRandomPlay:         0x1008FF99,
    XF86XK_Subtitle:                0x1008FF9A,
    XF86XK_AudioCycleTrack:         0x1008FF9B,
    XF86XK_CycleAngle:              0x1008FF9C,
    XF86XK_FrameBack:               0x1008FF9D,
    XF86XK_FrameForward:            0x1008FF9E,
    XF86XK_Time:                    0x1008FF9F,
    XF86XK_Select:                  0x1008FFA0,
    XF86XK_View:                    0x1008FFA1,
    XF86XK_TopMenu:                 0x1008FFA2,
    XF86XK_Red:                     0x1008FFA3,
    XF86XK_Green:                   0x1008FFA4,
    XF86XK_Yellow:                  0x1008FFA5,
    XF86XK_Blue:                    0x1008FFA6,
    XF86XK_Suspend:                 0x1008FFA7,
    XF86XK_Hibernate:               0x1008FFA8,
    XF86XK_TouchpadToggle:          0x1008FFA9,
    XF86XK_TouchpadOn:              0x1008FFB0,
    XF86XK_TouchpadOff:             0x1008FFB1,
    XF86XK_AudioMicMute:            0x1008FFB2,
    XF86XK_Switch_VT_1:             0x1008FE01,
    XF86XK_Switch_VT_2:             0x1008FE02,
    XF86XK_Switch_VT_3:             0x1008FE03,
    XF86XK_Switch_VT_4:             0x1008FE04,
    XF86XK_Switch_VT_5:             0x1008FE05,
    XF86XK_Switch_VT_6:             0x1008FE06,
    XF86XK_Switch_VT_7:             0x1008FE07,
    XF86XK_Switch_VT_8:             0x1008FE08,
    XF86XK_Switch_VT_9:             0x1008FE09,
    XF86XK_Switch_VT_10:            0x1008FE0A,
    XF86XK_Switch_VT_11:            0x1008FE0B,
    XF86XK_Switch_VT_12:            0x1008FE0C,
    XF86XK_Ungrab:                  0x1008FE20,
    XF86XK_ClearGrab:               0x1008FE21,
    XF86XK_Next_VMode:              0x1008FE22,
    XF86XK_Prev_VMode:              0x1008FE23,
    XF86XK_LogWindowTree:           0x1008FE24,
    XF86XK_LogGrabInfo:             0x1008FE25,
};
