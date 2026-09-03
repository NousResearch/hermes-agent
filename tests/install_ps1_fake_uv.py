"""Fake uv executable used by behavioral Windows installer tests."""

from __future__ import annotations

from pathlib import Path
import subprocess


_FAKE_UV = r'''
using System;
using System.IO;
using System.Linq;

public static class FakeUv {
    public static int Main(string[] args) {
        if (Path.GetFileName(Environment.GetCommandLineArgs()[0]).Equals(
                "python.exe", StringComparison.OrdinalIgnoreCase)) {
            Console.WriteLine(Environment.GetEnvironmentVariable("FAKE_PYTHON_VERSION")
                ?? "Python 3.11.0");
            return 0;
        }

        File.AppendAllText(Environment.GetEnvironmentVariable("FAKE_UV_LOG"),
            string.Join(" ", args) + Environment.NewLine);

        if (args.Length >= 2 && args[0] == "python" && args[1] == "find") {
            string findDelayMs = Environment.GetEnvironmentVariable(
                "FAKE_UV_FIND_DELAY_MS");
            if (!string.IsNullOrEmpty(findDelayMs)) {
                System.Threading.Thread.Sleep(int.Parse(findDelayMs));
            }
            string stderrBytes = Environment.GetEnvironmentVariable(
                "FAKE_UV_FIND_STDERR_BYTES");
            if (!string.IsNullOrEmpty(stderrBytes)) {
                Console.Error.Write(new string('x', int.Parse(stderrBytes)));
            }
            bool managed = args.Contains("--managed-python");
            string availableVersion = Environment.GetEnvironmentVariable(
                "FAKE_MANAGED_PYTHON_VERSION");
            if (managed && !string.IsNullOrEmpty(availableVersion)
                    && (args.Length < 3 || args[2] != availableVersion)) {
                return 1;
            }
            Console.WriteLine(Environment.GetEnvironmentVariable(
                managed ? "FAKE_MANAGED_PYTHON" : "FAKE_THIRD_PARTY_PYTHON"));
            return 0;
        }
        if (args.Length >= 2 && args[0] == "python" && args[1] == "install") {
            return 0;
        }
        if (args.Length >= 2 && args[0] == "venv" && args[1] == "venv") {
            string forcedExit = Environment.GetEnvironmentVariable("FAKE_UV_VENV_EXIT");
            if (!string.IsNullOrEmpty(forcedExit)) {
                return int.Parse(forcedExit);
            }
            string managedPython = Environment.GetEnvironmentVariable("FAKE_MANAGED_PYTHON");
            int pythonAt = Array.IndexOf(args, "--python");
            bool correctPython = pythonAt >= 0 && pythonAt + 1 < args.Length
                && string.Equals(args[pythonAt + 1], managedPython,
                    StringComparison.OrdinalIgnoreCase);
            if (!correctPython || !args.Contains("--managed-python")
                    || !args.Contains("--no-python-downloads")) {
                return 42;
            }
            string scripts = Path.Combine(Environment.CurrentDirectory, "venv", "Scripts");
            Directory.CreateDirectory(scripts);
            File.Copy(managedPython, Path.Combine(scripts, "python.exe"), true);
            return 0;
        }
        return 2;
    }
}
'''


def compile_fake_uv(powershell: str, output: Path) -> None:
    source = output.with_suffix(".cs")
    source.write_text(_FAKE_UV, encoding="utf-8")
    compile_script = output.with_name("compile-fake-uv.ps1")
    compile_script.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -Path $Source -OutputAssembly $Output "
        "-OutputType ConsoleApplication\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(compile_script),
            "-Source",
            str(source),
            "-Output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
