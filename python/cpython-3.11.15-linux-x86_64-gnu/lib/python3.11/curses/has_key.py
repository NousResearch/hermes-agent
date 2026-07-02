
#
# Emulation of has_key() function for platforms that don't use ncurses
#

import _curses

# Table mapping curses keys to the terminfo capability name

_capability_names = {
    _curses.KEY_A1: 'ka1',
    _curses.KEY_A3: 'ka3',
    _curses.KEY_B2: 'kb2',
    _curses.KEY_BACKSPACE: 'kbs',
    _curses.KEY_BEG: 'kbeg',
    _curses.KEY_BTAB: 'kcbt',
    _curses.KEY_C1: 'kc1',
    _curses.KEY_C3: 'kc3',
    _curses.KEY_CANCEL: 'kcan',
    _curses.KEY_CATAB: 'ktbc',
    _curses.KEY_CLEAR: 'kclr',
    _curses.KEY_CLOSE: 'kclo',
    _curses.KEY_COMMAND: 'kcmd',
    _curses.KEY_COPY: 'kcpy',
    _curses.KEY_CREATE: 'kcrt',
    _curses.KEY_CTAB: 'kctab',
    _curses.KEY_DC: 'kdch1',
    _curses.KEY_DL: 'kdl1',
    _curses.KEY_DOWN: 'kcud1',
    _curses.KEY_EIC: 'krmir',
    _curses.KEY_END: 'kend',
    _curses.KEY_ENTER: 'kent',
    _curses.KEY_EOL: 'kel',
    _curses.KEY_EOS: 'ked',
    _curses.KEY_EXIT: 'kext',
    _curses.KEY_F0: 'kf0',
    _curses.KEY_F1: 'kf1',
    _curses.KEY_F10: 'kf10',
    _curses.KEY_F11: 'kf11',
    _curses.KEY_F12: 'kf12',
    _curses.KEY_F13: 'kf13',
    _curses.KEY_F14: 'kf14',
    _curses.KEY_F15: 'kf15',
    _curses.KEY_F16: 'kf16',
    _curses.KEY_F17: 'kf17',
    _curses.KEY_F18: 'kf18',
    _curses.KEY_F19: 'kf19',
    _curses.KEY_F2: 'kf2',
    _curses.KEY_F20: 'kf20',
    _curses.KEY_F21: 'kf21',
    _curses.KEY_F22: 'kf22',
    _curses.KEY_F23: 'kf23',
    _curses.KEY_F24: 'kf24',
    _curses.KEY_F25: 'kf25',
    _curses.KEY_F26: 'kf26',
    _curses.KEY_F27: 'kf27',
    _curses.KEY_F28: 'kf28',
    _curses.KEY_F29: 'kf29',
    _curses.KEY_F3: 'kf3',
    _curses.KEY_F30: 'kf30',
    _curses.KEY_F31: 'kf31',
    _curses.KEY_F32: 'kf32',
    _curses.KEY_F33: 'kf33',
    _curses.KEY_F34: 'kf34',
    _curses.KEY_F35: 'kf35',
    _curses.KEY_F36: 'kf36',
    _curses.KEY_F37: 'kf37',
    _curses.KEY_F38: 'kf38',
    _curses.KEY_F39: 'kf39',
    _curses.KEY_F4: 'kf4',
    _curses.KEY_F40: 'kf40',
    _curses.KEY_F41: 'kf41',
    _curses.KEY_F42: 'kf42',
    _curses.KEY_F43: 'kf43',
    _curses.KEY_F44: 'kf44',
    _curses.KEY_F45: 'kf45',
    _curses.KEY_F46: 'kf46',
    _curses.KEY_F47: 'kf47',
    _curses.KEY_F48: 'kf48',
    _curses.KEY_F49: 'kf49',
    _curses.KEY_F5: 'kf5',
    _curses.KEY_F50: 'kf50',
    _curses.KEY_F51: 'kf51',
    _curses.KEY_F52: 'kf52',
    _curses.KEY_F53: 'kf53',
    _curses.KEY_F54: 'kf54',
    _curses.KEY_F55: 'kf55',
    _curses.KEY_F56: 'kf56',
    _curses.KEY_F57: 'kf57',
    _curses.KEY_F58: 'kf58',
    _curses.KEY_F59: 'kf59',
    _curses.KEY_F6: 'kf6',
    _curses.KEY_F60: 'kf60',
    _curses.KEY_F61: 'kf61',
    _curses.KEY_F62: 'kf62',
    _curses.KEY_F63: 'kf63',
    _curses.KEY_F7: 'kf7',
    _curses.KEY_F8: 'kf8',
    _curses.KEY_F9: 'kf9',
    _curses.KEY_FIND: 'kfnd',
    _curses.KEY_HELP: 'khlp',
    _curses.KEY_HOME: 'khome',
    _curses.KEY_IC: 'kich1',
    _curses.KEY_IL: 'kil1',
    _curses.KEY_LEFT: 'kcub1',
    _curses.KEY_LL: 'kll',
    _curses.KEY_MARK: 'kmrk',
    _curses.KEY_MESSAGE: 'kmsg',
    _curses.KEY_MOVE: 'kmov',
    _curses.KEY_NEXT: 'knxt',
    _curses.KEY_NPAGE: 'knp',
    _curses.KEY_OPEN: 'kopn',
    _curses.KEY_OPTIONS: 'kopt',
    _curses.KEY_PPAGE: 'kpp',
    _curses.KEY_PREVIOUS: 'kprv',
    _curses.KEY_PRINT: 'kprt',
    _curses.KEY_REDO: 'krdo',
    _curses.KEY_REFERENCE: 'kref',
    _curses.KEY_REFRESH: 'krfr',
    _curses.KEY_REPLACE: 'krpl',
    _curses.KEY_RESTART: 'krst',
    _curses.KEY_RESUME: 'kres',
    _curses.KEY_RIGHT: 'kcuf1',
    _curses.KEY_SAVE: 'ksav',
    _curses.KEY_SBEG: 'kBEG',
    _curses.KEY_SCANCEL: 'kCAN',
    _curses.KEY_SCOMMAND: 'kCMD',
    _curses.KEY_SCOPY: 'kCPY',
    _curses.KEY_SCREATE: 'kCRT',
    _curses.KEY_SDC: 'kDC',
    _curses.KEY_SDL: 'kDL',
    _curses.KEY_SELECT: 'kslt',
    _curses.KEY_SEND: 'kEND',
    _curses.KEY_SEOL: 'kEOL',
    _curses.KEY_SEXIT: 'kEXT',
    _curses.KEY_SF: 'kind',
    _curses.KEY_SFIND: 'kFND',
    _curses.KEY_SHELP: 'kHLP',
    _curses.KEY_SHOME: 'kHOM',
    _curses.KEY_SIC: 'kIC',
    _curses.KEY_SLEFT: 'kLFT',
    _curses.KEY_SMESSAGE: 'kMSG',
    _curses.KEY_SMOVE: 'kMOV',
    _curses.KEY_SNEXT: 'kNXT',
    _curses.KEY_SOPTIONS: 'kOPT',
    _curses.KEY_SPREVIOUS: 'kPRV',
    _curses.KEY_SPRINT: 'kPRT',
    _curses.KEY_SR: 'kri',
    _curses.KEY_SREDO: 'kRDO',
    _curses.KEY_SREPLACE: 'kRPL',
    _curses.KEY_SRIGHT: 'kRIT',
    _curses.KEY_SRSUME: 'kRES',
    _curses.KEY_SSAVE: 'kSAV',
    _curses.KEY_SSUSPEND: 'kSPD',
    _curses.KEY_STAB: 'khts',
    _curses.KEY_SUNDO: 'kUND',
    _curses.KEY_SUSPEND: 'kspd',
    _curses.KEY_UNDO: 'kund',
    _curses.KEY_UP: 'kcuu1'
    }

def has_key(ch):
    if isinstance(ch, str):
        ch = ord(ch)

    # Figure out the correct capability name for the keycode.
    capability_name = _capability_names.get(ch)
    if capability_name is None:
        return False

    #Check the current terminal description for that capability;
    #if present, return true, else return false.
    if _curses.tigetstr( capability_name ):
        return True
    else:
        return False

if __name__ == '__main__':
    # Compare the output of this implementation and the ncurses has_key,
    # on platforms where has_key is already available
    try:
        L = []
        _curses.initscr()
        for key in _capability_names.keys():
            system = _curses.has_key(key)
            python = has_key(key)
            if system != python:
                L.append( 'Mismatch for key %s, system=%i, Python=%i'
                          % (_curses.keyname( key ), system, python) )
    finally:
        _curses.endwin()
        for i in L: print(i)
