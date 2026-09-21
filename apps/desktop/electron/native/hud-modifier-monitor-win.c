// Raw Input with INPUTSINK observes without hooks, grabs, or suppressing keys.
#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0601
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include "hud-modifier-gesture.h"

static HudModifierGesture gesture;
static bool keys[256]; // Current physical state, not a log or character buffer.
static unsigned buttons;
static HANDLE output;
static void Emit(const char *line) {
  DWORD written = 0, size = (DWORD)strlen(line);
  if (!WriteFile(output, line, size, &written, NULL) || written != size) ExitProcess(0);
}
static void Unavailable(void) { Emit("{\"type\":\"error\",\"code\":\"unavailable\"}\n"); }
static bool Target(unsigned key) { return key == VK_LCONTROL || key == VK_RCONTROL || key == VK_LMENU; }
static bool MouseKey(unsigned key) {
  return key == VK_LBUTTON || key == VK_RBUTTON || key == VK_MBUTTON || key == VK_XBUTTON1 || key == VK_XBUTTON2;
}
static void Update(bool interrupted) {
  unsigned first = keys[VK_LCONTROL] + keys[VK_RCONTROL];
  // Right Alt is AltGr on many layouts. Never summon for the generated Ctrl+Alt
  // pair of an AltGr tap; Ctrl on either side + left Alt remains available.
  unsigned modifiers = (first ? 1u : 0u) | (keys[VK_LMENU] ? 2u : 0u) | (first > 1 ? 4u : 0u);
  bool held = buttons != 0;
  for (unsigned key = 0; key < 256; key++) {
    if (!Target(key) && !MouseKey(key) && key != VK_CONTROL && key != VK_MENU && key != VK_SHIFT) held |= keys[key];
  }
  if (HudModifierUpdate(&gesture, modifiers, held, interrupted, GetTickCount64())) Emit("{\"type\":\"summon\"}\n");
}
static void Snapshot(void) {
  for (unsigned key = 0; key < 256; key++) keys[key] = (GetAsyncKeyState((int)key) & 0x8000) != 0;
  buttons = keys[VK_LBUTTON] | (keys[VK_RBUTTON] << 1) | (keys[VK_MBUTTON] << 2)
    | (keys[VK_XBUTTON1] << 3) | (keys[VK_XBUTTON2] << 4);
  gesture = (HudModifierGesture){0};
  Update(true);
}
static LRESULT CALLBACK Observe(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
  if (message == WM_INPUT_DEVICE_CHANGE) {
    Snapshot();
  } else if (message == WM_INPUT) {
    RAWINPUT raw;
    UINT size = sizeof(raw);
    if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER)) == (UINT)-1) {
      Unavailable(); PostQuitMessage(3);
    } else if (raw.header.dwType == RIM_TYPEKEYBOARD) {
      RAWKEYBOARD *key = &raw.data.keyboard;
      unsigned code = key->VKey;
      if (code == VK_CONTROL) code = (key->Flags & RI_KEY_E0) ? VK_RCONTROL : VK_LCONTROL;
      if (code == VK_MENU) code = (key->Flags & RI_KEY_E0) ? VK_RMENU : VK_LMENU;
      if (code == VK_SHIFT) code = MapVirtualKeyW(key->MakeCode, MAPVK_VSC_TO_VK_EX);
      bool down = !(key->Flags & RI_KEY_BREAK);
      bool interrupted = code >= 256 || !Target(code);
      if (code < 256) {
        interrupted |= keys[code] == down; // Repeat, duplicate, or missed physical transition.
        keys[code] = down;
      }
      Update(interrupted);
    } else if (raw.header.dwType == RIM_TYPEMOUSE) {
      unsigned flags = raw.data.mouse.usButtonFlags;
      for (unsigned button = 0; button < 5; button++) {
        if (flags & (1u << (button * 2))) buttons |= 1u << button;
        if (flags & (2u << (button * 2))) buttons &= ~(1u << button);
      }
      Update(flags != 0 || buttons != 0); // Wheel, click, or motion while dragging.
    }
    // DefWindowProc performs Raw Input cleanup, not input consumption.
  } else if (message == WM_CLOSE) {
    DestroyWindow(window);
    return 0;
  } else if (message == WM_DESTROY) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcW(window, message, wParam, lParam);
}
static DWORD WINAPI WatchParent(void *context) {
  char byte;
  DWORD count;
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  while (ReadFile(input, &byte, 1, &count, NULL) && count) {}
  PostMessageW((HWND)context, WM_CLOSE, 0, 0);
  return 0;
}
int main(int argc, char **argv) {
  output = GetStdHandle(STD_OUTPUT_HANDLE);
  // Node supplies an anonymous/named pipe. Never block the window's input loop
  // behind a stalled parent; full-pipe writes fail and end this helper.
  DWORD mode = PIPE_NOWAIT;
  if (GetFileType(output) != FILE_TYPE_PIPE || !SetNamedPipeHandleState(output, &mode, NULL, NULL)) return 3;
  bool check = argc == 2 && strcmp(argv[1], "--check") == 0;
  if (argc > 1 && !check && !(argc == 2 && strcmp(argv[1], "--request-permission") == 0)) { Unavailable(); return 64; }
  HINSTANCE instance = GetModuleHandleW(NULL);
  WNDCLASSW definition = {0};
  definition.lpfnWndProc = Observe; definition.hInstance = instance;
  definition.lpszClassName = L"HermesHudModifierMonitor";
  if (!RegisterClassW(&definition)) { Unavailable(); return 3; }
  HWND window = CreateWindowExW(0, definition.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, instance, NULL);
  if (!window) { Unavailable(); return 3; }
  RAWINPUTDEVICE devices[] = {
    { 1, 6, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, window },
    { 1, 2, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, window }
  };
  if (!RegisterRawInputDevices(devices, 2, sizeof(RAWINPUTDEVICE))) { Unavailable(); DestroyWindow(window); return 3; }
  Snapshot();
  HANDLE parent = CreateThread(NULL, 0, WatchParent, window, 0, NULL);
  if (!parent) { Unavailable(); DestroyWindow(window); return 3; }
  Emit("{\"type\":\"ready\"}\n");
  MSG message;
  int result = 0;
  while (!check && (result = GetMessageW(&message, NULL, 0, 0)) > 0) {
    TranslateMessage(&message); DispatchMessageW(&message);
  }
  if (result < 0) Unavailable();
  DestroyWindow(window);
  CancelSynchronousIo(parent);
  CloseHandle(parent);
  UnregisterClassW(definition.lpszClassName, instance);
  return result < 0 ? 3 : 0;
}
