/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2018 The noVNC authors
 * Licensed under MPL 2.0 or any later version (see LICENSE.txt)
 */

import KeyTable from "./keysym.js";

/*
 * Mapping between HTML key values and VNC/X11 keysyms for "special"
 * keys that cannot be handled via their Unicode codepoint.
 *
 * See https://www.w3.org/TR/uievents-key/ for possible values.
 */

const DOMKeyTable = {};

function addStandard(key, standard) {
    if (standard === undefined) throw new Error("Undefined keysym for key \"" + key + "\"");
    if (key in DOMKeyTable) throw new Error("Duplicate entry for key \"" + key + "\"");
    DOMKeyTable[key] = [standard, standard, standard, standard];
}

function addLeftRight(key, left, right) {
    if (left === undefined) throw new Error("Undefined keysym for key \"" + key + "\"");
    if (right === undefined) throw new Error("Undefined keysym for key \"" + key + "\"");
    if (key in DOMKeyTable) throw new Error("Duplicate entry for key \"" + key + "\"");
    DOMKeyTable[key] = [left, left, right, left];
}

function addNumpad(key, standard, numpad) {
    if (standard === undefined) throw new Error("Undefined keysym for key \"" + key + "\"");
    if (numpad === undefined) throw new Error("Undefined keysym for key \"" + key + "\"");
    if (key in DOMKeyTable) throw new Error("Duplicate entry for key \"" + key + "\"");
    DOMKeyTable[key] = [standard, standard, standard, numpad];
}

// 3.2. Modifier Keys

addLeftRight("Alt", KeyTable.XK_Alt_L, KeyTable.XK_Alt_R);
addStandard("AltGraph", KeyTable.XK_ISO_Level3_Shift);
addStandard("CapsLock", KeyTable.XK_Caps_Lock);
addLeftRight("Control", KeyTable.XK_Control_L, KeyTable.XK_Control_R);
// - Fn
// - FnLock
addLeftRight("Meta", KeyTable.XK_Super_L, KeyTable.XK_Super_R);
addStandard("NumLock", KeyTable.XK_Num_Lock);
addStandard("ScrollLock", KeyTable.XK_Scroll_Lock);
addLeftRight("Shift", KeyTable.XK_Shift_L, KeyTable.XK_Shift_R);
// - Symbol
// - SymbolLock
// - Hyper
// - Super

// 3.3. Whitespace Keys

addNumpad("Enter", KeyTable.XK_Return, KeyTable.XK_KP_Enter);
addStandard("Tab", KeyTable.XK_Tab);
addNumpad(" ", KeyTable.XK_space, KeyTable.XK_KP_Space);

// 3.4. Navigation Keys

addNumpad("ArrowDown", KeyTable.XK_Down, KeyTable.XK_KP_Down);
addNumpad("ArrowLeft", KeyTable.XK_Left, KeyTable.XK_KP_Left);
addNumpad("ArrowRight", KeyTable.XK_Right, KeyTable.XK_KP_Right);
addNumpad("ArrowUp", KeyTable.XK_Up, KeyTable.XK_KP_Up);
addNumpad("End", KeyTable.XK_End, KeyTable.XK_KP_End);
addNumpad("Home", KeyTable.XK_Home, KeyTable.XK_KP_Home);
addNumpad("PageDown", KeyTable.XK_Next, KeyTable.XK_KP_Next);
addNumpad("PageUp", KeyTable.XK_Prior, KeyTable.XK_KP_Prior);

// 3.5. Editing Keys

addStandard("Backspace", KeyTable.XK_BackSpace);
// Browsers send "Clear" for the numpad 5 without NumLock because
// Windows uses VK_Clear for that key. But Unix expects KP_Begin for
// that scenario.
addNumpad("Clear", KeyTable.XK_Clear, KeyTable.XK_KP_Begin);
addStandard("Copy", KeyTable.XF86XK_Copy);
// - CrSel
addStandard("Cut", KeyTable.XF86XK_Cut);
addNumpad("Delete", KeyTable.XK_Delete, KeyTable.XK_KP_Delete);
// - EraseEof
// - ExSel
addNumpad("Insert", KeyTable.XK_Insert, KeyTable.XK_KP_Insert);
addStandard("Paste", KeyTable.XF86XK_Paste);
addStandard("Redo", KeyTable.XK_Redo);
addStandard("Undo", KeyTable.XK_Undo);

// 3.6. UI Keys

// - Accept
// - Again (could just be XK_Redo)
// - Attn
addStandard("Cancel", KeyTable.XK_Cancel);
addStandard("ContextMenu", KeyTable.XK_Menu);
addStandard("Escape", KeyTable.XK_Escape);
addStandard("Execute", KeyTable.XK_Execute);
addStandard("Find", KeyTable.XK_Find);
addStandard("Help", KeyTable.XK_Help);
addStandard("Pause", KeyTable.XK_Pause);
// - Play
// - Props
addStandard("Select", KeyTable.XK_Select);
addStandard("ZoomIn", KeyTable.XF86XK_ZoomIn);
addStandard("ZoomOut", KeyTable.XF86XK_ZoomOut);

// 3.7. Device Keys

addStandard("BrightnessDown", KeyTable.XF86XK_MonBrightnessDown);
addStandard("BrightnessUp", KeyTable.XF86XK_MonBrightnessUp);
addStandard("Eject", KeyTable.XF86XK_Eject);
addStandard("LogOff", KeyTable.XF86XK_LogOff);
addStandard("Power", KeyTable.XF86XK_PowerOff);
addStandard("PowerOff", KeyTable.XF86XK_PowerDown);
addStandard("PrintScreen", KeyTable.XK_Print);
addStandard("Hibernate", KeyTable.XF86XK_Hibernate);
addStandard("Standby", KeyTable.XF86XK_Standby);
addStandard("WakeUp", KeyTable.XF86XK_WakeUp);

// 3.8. IME and Composition Keys

addStandard("AllCandidates", KeyTable.XK_MultipleCandidate);
addStandard("Alphanumeric", KeyTable.XK_Eisu_toggle);
addStandard("CodeInput", KeyTable.XK_Codeinput);
addStandard("Compose", KeyTable.XK_Multi_key);
addStandard("Convert", KeyTable.XK_Henkan);
// - Dead
// - FinalMode
addStandard("GroupFirst", KeyTable.XK_ISO_First_Group);
addStandard("GroupLast", KeyTable.XK_ISO_Last_Group);
addStandard("GroupNext", KeyTable.XK_ISO_Next_Group);
addStandard("GroupPrevious", KeyTable.XK_ISO_Prev_Group);
// - ModeChange (XK_Mode_switch is often used for AltGr)
// - NextCandidate
addStandard("NonConvert", KeyTable.XK_Muhenkan);
addStandard("PreviousCandidate", KeyTable.XK_PreviousCandidate);
// - Process
addStandard("SingleCandidate", KeyTable.XK_SingleCandidate);
addStandard("HangulMode", KeyTable.XK_Hangul);
addStandard("HanjaMode", KeyTable.XK_Hangul_Hanja);
addStandard("JunjaMode", KeyTable.XK_Hangul_Jeonja);
addStandard("Eisu", KeyTable.XK_Eisu_toggle);
addStandard("Hankaku", KeyTable.XK_Hankaku);
addStandard("Hiragana", KeyTable.XK_Hiragana);
addStandard("HiraganaKatakana", KeyTable.XK_Hiragana_Katakana);
addStandard("KanaMode", KeyTable.XK_Kana_Shift); // could also be _Kana_Lock
addStandard("KanjiMode", KeyTable.XK_Kanji);
addStandard("Katakana", KeyTable.XK_Katakana);
addStandard("Romaji", KeyTable.XK_Romaji);
addStandard("Zenkaku", KeyTable.XK_Zenkaku);
addStandard("ZenkakuHankaku", KeyTable.XK_Zenkaku_Hankaku);

// 3.9. General-Purpose Function Keys

addStandard("F1", KeyTable.XK_F1);
addStandard("F2", KeyTable.XK_F2);
addStandard("F3", KeyTable.XK_F3);
addStandard("F4", KeyTable.XK_F4);
addStandard("F5", KeyTable.XK_F5);
addStandard("F6", KeyTable.XK_F6);
addStandard("F7", KeyTable.XK_F7);
addStandard("F8", KeyTable.XK_F8);
addStandard("F9", KeyTable.XK_F9);
addStandard("F10", KeyTable.XK_F10);
addStandard("F11", KeyTable.XK_F11);
addStandard("F12", KeyTable.XK_F12);
addStandard("F13", KeyTable.XK_F13);
addStandard("F14", KeyTable.XK_F14);
addStandard("F15", KeyTable.XK_F15);
addStandard("F16", KeyTable.XK_F16);
addStandard("F17", KeyTable.XK_F17);
addStandard("F18", KeyTable.XK_F18);
addStandard("F19", KeyTable.XK_F19);
addStandard("F20", KeyTable.XK_F20);
addStandard("F21", KeyTable.XK_F21);
addStandard("F22", KeyTable.XK_F22);
addStandard("F23", KeyTable.XK_F23);
addStandard("F24", KeyTable.XK_F24);
addStandard("F25", KeyTable.XK_F25);
addStandard("F26", KeyTable.XK_F26);
addStandard("F27", KeyTable.XK_F27);
addStandard("F28", KeyTable.XK_F28);
addStandard("F29", KeyTable.XK_F29);
addStandard("F30", KeyTable.XK_F30);
addStandard("F31", KeyTable.XK_F31);
addStandard("F32", KeyTable.XK_F32);
addStandard("F33", KeyTable.XK_F33);
addStandard("F34", KeyTable.XK_F34);
addStandard("F35", KeyTable.XK_F35);
// - Soft1...

// 3.10. Multimedia Keys

// - ChannelDown
// - ChannelUp
addStandard("Close", KeyTable.XF86XK_Close);
addStandard("MailForward", KeyTable.XF86XK_MailForward);
addStandard("MailReply", KeyTable.XF86XK_Reply);
addStandard("MailSend", KeyTable.XF86XK_Send);
// - MediaClose
addStandard("MediaFastForward", KeyTable.XF86XK_AudioForward);
addStandard("MediaPause", KeyTable.XF86XK_AudioPause);
addStandard("MediaPlay", KeyTable.XF86XK_AudioPlay);
// - MediaPlayPause
addStandard("MediaRecord", KeyTable.XF86XK_AudioRecord);
addStandard("MediaRewind", KeyTable.XF86XK_AudioRewind);
addStandard("MediaStop", KeyTable.XF86XK_AudioStop);
addStandard("MediaTrackNext", KeyTable.XF86XK_AudioNext);
addStandard("MediaTrackPrevious", KeyTable.XF86XK_AudioPrev);
addStandard("New", KeyTable.XF86XK_New);
addStandard("Open", KeyTable.XF86XK_Open);
addStandard("Print", KeyTable.XK_Print);
addStandard("Save", KeyTable.XF86XK_Save);
addStandard("SpellCheck", KeyTable.XF86XK_Spell);

// 3.11. Multimedia Numpad Keys

// - Key11
// - Key12

// 3.12. Audio Keys

// - AudioBalanceLeft
// - AudioBalanceRight
// - AudioBassBoostDown
// - AudioBassBoostToggle
// - AudioBassBoostUp
// - AudioFaderFront
// - AudioFaderRear
// - AudioSurroundModeNext
// - AudioTrebleDown
// - AudioTrebleUp
addStandard("AudioVolumeDown", KeyTable.XF86XK_AudioLowerVolume);
addStandard("AudioVolumeUp", KeyTable.XF86XK_AudioRaiseVolume);
addStandard("AudioVolumeMute", KeyTable.XF86XK_AudioMute);
// - MicrophoneToggle
// - MicrophoneVolumeDown
// - MicrophoneVolumeUp
addStandard("MicrophoneVolumeMute", KeyTable.XF86XK_AudioMicMute);

// 3.13. Speech Keys

// - SpeechCorrectionList
// - SpeechInputToggle

// 3.14. Application Keys

addStandard("LaunchApplication1", KeyTable.XF86XK_MyComputer);
addStandard("LaunchApplication2", KeyTable.XF86XK_Calculator);
addStandard("LaunchCalendar", KeyTable.XF86XK_Calendar);
// - LaunchContacts
addStandard("LaunchMail", KeyTable.XF86XK_Mail);
addStandard("LaunchMediaPlayer", KeyTable.XF86XK_AudioMedia);
addStandard("LaunchMusicPlayer", KeyTable.XF86XK_Music);
addStandard("LaunchPhone", KeyTable.XF86XK_Phone);
addStandard("LaunchScreenSaver", KeyTable.XF86XK_ScreenSaver);
addStandard("LaunchSpreadsheet", KeyTable.XF86XK_Excel);
addStandard("LaunchWebBrowser", KeyTable.XF86XK_WWW);
addStandard("LaunchWebCam", KeyTable.XF86XK_WebCam);
addStandard("LaunchWordProcessor", KeyTable.XF86XK_Word);

// 3.15. Browser Keys

addStandard("BrowserBack", KeyTable.XF86XK_Back);
addStandard("BrowserFavorites", KeyTable.XF86XK_Favorites);
addStandard("BrowserForward", KeyTable.XF86XK_Forward);
addStandard("BrowserHome", KeyTable.XF86XK_HomePage);
addStandard("BrowserRefresh", KeyTable.XF86XK_Refresh);
addStandard("BrowserSearch", KeyTable.XF86XK_Search);
addStandard("BrowserStop", KeyTable.XF86XK_Stop);

// 3.16. Mobile Phone Keys

// - A whole bunch...

// 3.17. TV Keys

// - A whole bunch...

// 3.18. Media Controller Keys

// - A whole bunch...
addStandard("Dimmer", KeyTable.XF86XK_BrightnessAdjust);
addStandard("MediaAudioTrack", KeyTable.XF86XK_AudioCycleTrack);
addStandard("RandomToggle", KeyTable.XF86XK_AudioRandomPlay);
addStandard("SplitScreenToggle", KeyTable.XF86XK_SplitScreen);
addStandard("Subtitle", KeyTable.XF86XK_Subtitle);
addStandard("VideoModeNext", KeyTable.XF86XK_Next_VMode);

// Extra: Numpad

addNumpad("=", KeyTable.XK_equal, KeyTable.XK_KP_Equal);
addNumpad("+", KeyTable.XK_plus, KeyTable.XK_KP_Add);
addNumpad("-", KeyTable.XK_minus, KeyTable.XK_KP_Subtract);
addNumpad("*", KeyTable.XK_asterisk, KeyTable.XK_KP_Multiply);
addNumpad("/", KeyTable.XK_slash, KeyTable.XK_KP_Divide);
addNumpad(".", KeyTable.XK_period, KeyTable.XK_KP_Decimal);
addNumpad(",", KeyTable.XK_comma, KeyTable.XK_KP_Separator);
addNumpad("0", KeyTable.XK_0, KeyTable.XK_KP_0);
addNumpad("1", KeyTable.XK_1, KeyTable.XK_KP_1);
addNumpad("2", KeyTable.XK_2, KeyTable.XK_KP_2);
addNumpad("3", KeyTable.XK_3, KeyTable.XK_KP_3);
addNumpad("4", KeyTable.XK_4, KeyTable.XK_KP_4);
addNumpad("5", KeyTable.XK_5, KeyTable.XK_KP_5);
addNumpad("6", KeyTable.XK_6, KeyTable.XK_KP_6);
addNumpad("7", KeyTable.XK_7, KeyTable.XK_KP_7);
addNumpad("8", KeyTable.XK_8, KeyTable.XK_KP_8);
addNumpad("9", KeyTable.XK_9, KeyTable.XK_KP_9);

export default DOMKeyTable;
