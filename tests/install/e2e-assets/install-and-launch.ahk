#Requires AutoHotkey v2.0
#SingleInstance Force

; Drive the REAL Hermes-Setup.exe (Tauri bootstrap installer) window:
; click Install, wait for the install to finish, click Launch, and wait for
; the real Hermes.exe (Electron desktop) window to appear.
;
; Adapted from @ethernet8023's e2e/windows/install-hermes-desktop.ahk
; (PR #68183) -- same ImageSearch approach; the install-button template was
; re-captured from a live CI frame (the #68183 templates predated the
; installer UI restyle to the "[ INSTALL ]" bracket look and never matched).
;
; Robustness beyond the original -- every one earned from a real CI failure:
;   * Log() survives a missing stdout. GUI-subsystem AHK started via
;     Start-Process has no console, so FileAppend to '*' throws
;     "(6) The handle is invalid" -- that single throw killed attempt 1.
;   * We wait for a REAL-sized installer window before reading geometry.
;     ahk_exe matched a hidden 16x16 helper window first (attempt 2's
;     "Window found at ... w=16 h=16"), so relative-position math was
;     computed against a phantom rect.
;   * Clicks fall back to a SCREEN-fraction position when the template
;     doesn't match (the Tauri window is effectively full-screen on the
;     runner, so screen ~= window). Button center measured at ~0.50, 0.59.
;   * Install-finished has a second signal: "bootstrap complete" in
;     bootstrap-installer.log, so a Launch-template miss can't strand us.
;
; Args: [1] log path  [2] setup exe name  [3] bootstrap-installer.log path

logPath := A_Args.Length >= 1 ? A_Args[1] : "ahk.log"
setupExe := A_Args.Length >= 2 ? A_Args[2] : "Hermes-Setup.exe"
bootstrapLog := A_Args.Length >= 3 ? A_Args[3] : ""

Log(text) {
    msg := Format("[autohotkey] {}`n", text)
    ToolTip(text)
    try FileAppend(msg, '*')   ; stdout may not exist (no console)
    FileAppend(msg, logPath)
}

OnError(LogError)
LogError(err, mode) {
    Log(Format("Unhandled error: {}", err.Message))
    ExitApp(1)
    return -1
}

SetWorkingDir(A_ScriptDir)
CoordMode("Pixel", "Screen")
CoordMode("Mouse", "Screen")

ClickWithMarker(x, y) {
    Click(x, y)
    Sleep(10)
    MouseMove(30, 30)
    Log(Format("Clicked at {1}, {2}", x, y))
}

; Wait until an installer window exists AND has a real (non-phantom) size,
; then return its rect. ahk_exe can transiently match a hidden helper.
WaitForRealWindow(winTitle, timeoutMs) {
    deadline := A_TickCount + timeoutMs
    while (A_TickCount < deadline) {
        for hwnd in WinGetList(winTitle) {
            try {
                WinGetPos(&wx, &wy, &ww, &wh, "ahk_id " hwnd)
                if (ww > 400 && wh > 300) {
                    return { hwnd: hwnd, x: wx, y: wy, w: ww, h: wh }
                }
            } catch {
                continue
            }
        }
        Sleep(500)
    }
    throw Error(Format("no real-sized window matched {} within {}ms", winTitle, timeoutMs))
}

; Image search inside a rect. Returns true + center coords.
TryFindImage(rect, imageFile, &outX, &outY) {
    hBitmap := LoadPicture(imageFile)
    if !hBitmap {
        throw Error("LoadPicture failed: " imageFile)
    }
    bm := Buffer(32, 0)
    DllCall("GetObject", "Ptr", hBitmap, "Int", bm.Size, "Ptr", bm)
    width := NumGet(bm, 4, "Int")
    height := NumGet(bm, 8, "Int")
    if ImageSearch(&x, &y, rect.x, rect.y, rect.x + rect.w, rect.y + rect.h, Format("*20 {}", imageFile)) {
        outX := x + Floor(width / 2)
        outY := y + Floor(height / 2)
        return true
    }
    return false
}

BootstrapLogContains(needle) {
    global bootstrapLog
    if (bootstrapLog = "" or !FileExist(bootstrapLog)) {
        return false
    }
    try {
        f := FileOpen(bootstrapLog, "r-d")   ; read, share read+write
        if !f {
            return false
        }
        content := f.Read()
        f.Close()
        return InStr(content, needle) > 0
    } catch {
        return false
    }
}

installerWin := "ahk_exe " setupExe
appWin := "ahk_exe Hermes.exe"

; Button center as a fraction of the window rect (installer is ~full-screen
; on the runner). Measured from a live CI frame where the template match
; landed at screen (511,454) inside window x=64 y=34 w=896 h=659:
;   fx = (511-64)/896 = 0.50 ; fy = (454-34)/659 = 0.637
BTN_FX := 0.50
BTN_FY := 0.637

Log("Waiting for a real-sized installer window (" installerWin ") ...")
rect := WaitForRealWindow(installerWin, 90000)
Log(Format("Installer window: x={1} y={2} w={3} h={4}", rect.x, rect.y, rect.w, rect.h))
try {
    WinActivate("ahk_id " rect.hwnd)
    Sleep(500)
} catch {
    Log("WARNING: could not activate installer window")
}

; -- Step 1: click Install -----------------------------------------------
installClicked := false
deadline := A_TickCount + 30000
while (A_TickCount < deadline) {
    if TryFindImage(rect, A_ScriptDir "\install-button.png", &ix, &iy) {
        ClickWithMarker(ix, iy)
        Log("Install clicked (template match)")
        installClicked := true
        break
    }
    Sleep(500)
}
if !installClicked {
    ix := rect.x + Floor(rect.w * BTN_FX)
    iy := rect.y + Floor(rect.h * BTN_FY)
    ClickWithMarker(ix, iy)
    Log("FALLBACK: install template never matched; clicked window-relative position")
}

; -- Step 2: wait for the install to finish ------------------------------
; Primary: the "bootstrap complete" line in the installer's own log -- the
; authoritative done signal. Secondary: the Launch button template.
launchX := 0, launchY := 0
launchFound := false
complete := false
waitDeadline := A_TickCount + 1000 * 60 * 45
Log("Waiting for install to finish (bootstrap log or Launch template) ...")
while (A_TickCount < waitDeadline) {
    if BootstrapLogContains("bootstrap complete") {
        complete := true
        Log("bootstrap-installer.log reports completion")
        break
    }
    ; refresh the rect (window can move/resize between stages)
    try rect := WaitForRealWindow(installerWin, 2000)
    if TryFindImage(rect, A_ScriptDir "\launch-button.png", &launchX, &launchY) {
        launchFound := true
        Log("Install finished (Launch template visible)")
        break
    }
    Sleep(2000)
}
if (!launchFound and !complete) {
    throw Error("install did not finish within 45 minutes (no completion log line, no Launch button)")
}

; Give the button a moment to swap Install -> Launch after completion.
if (complete and !launchFound) {
    Sleep(2000)
    try rect := WaitForRealWindow(installerWin, 10000)
    if TryFindImage(rect, A_ScriptDir "\launch-button.png", &launchX, &launchY) {
        launchFound := true
        Log("Launch template matched after completion")
    }
}

; -- Step 3: click Launch -- the hand-off under test ----------------------
if launchFound {
    ClickWithMarker(launchX, launchY)
    Log("Launch clicked (template)")
} else {
    lx := rect.x + Floor(rect.w * BTN_FX)
    ly := rect.y + Floor(rect.h * BTN_FY)
    ClickWithMarker(lx, ly)
    Log("FALLBACK: clicked Launch at window-relative position")
}
Log("Launch clicked; waiting for the Hermes desktop app window")

; The installer spawns Hermes.exe detached and exits itself.
try {
    WinWait(appWin, , 120)
} catch {
    throw Error("Hermes.exe window did not appear within 120s of clicking Launch")
}
WinGetPos(&ax, &ay, &aw, &ah, appWin)
Log(Format("App window appeared at x={1} y={2} w={3} h={4}", ax, ay, aw, ah))

Sleep(8000)   ; let the renderer paint (recorded as proof)
Log("done")
ExitApp(0)
