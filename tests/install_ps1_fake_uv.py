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
        File.AppendAllText(Environment.GetEnvironmentVariable("FAKE_UV_LOG"),
            string.Join(" ", args) + Environment.NewLine);

        if (args.Length >= 2 && args[0] == "python" && args[1] == "find") {
            bool managed = args.Contains("--managed-python");
            Console.WriteLine(Environment.GetEnvironmentVariable(
                managed ? "FAKE_MANAGED_PYTHON" : "FAKE_THIRD_PARTY_PYTHON"));
            return 0;
        }
        if (args.Length >= 2 && args[0] == "python" && args[1] == "install") {
            return 0;
        }
        if (args.Length >= 2 && args[0] == "venv" && args[1] == "venv") {
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
            File.WriteAllText(Path.Combine(scripts, "python.exe"), "fake");
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
