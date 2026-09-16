// Starts the tray watchdog detached from any parent process tree.
// WScript's Run(window=0, wait=false) orphans the child immediately, so the
// watchdog survives desktop-app restarts (anything started from a shell that
// is a child of the app tree dies with it - the classic "autostart silently
// stops working" bug). All paths resolve from this file's own folder.
// ASCII content only: cscript reads .js as ANSI/GBK, never UTF-8.
var fso = new ActiveXObject("Scripting.FileSystemObject");
var dir = fso.GetParentFolderName(WScript.ScriptFullName);
var pythonw = dir + "/venv/Scripts/pythonw.exe";
if (!fso.FileExists(pythonw)) {
    WScript.Echo("tray venv missing: " + pythonw + "  (run install_tray.ps1 first)");
    WScript.Quit(1);
}
var ws = new ActiveXObject("WScript.Shell");
ws.CurrentDirectory = dir;
ws.Run('"' + pythonw + '" "' + dir + "/tray_watchdog.py\"", 0, false);
WScript.Echo("tray watchdog launched (hidden, orphan)");
