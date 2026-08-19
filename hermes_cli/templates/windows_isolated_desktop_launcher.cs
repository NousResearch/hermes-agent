using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

internal static class {{CLASS_NAME}}
{
    private const string LauncherDirectory = @"{{LAUNCHER_DIRECTORY}}";
    private const string SharedHermesExe = @"{{SHARED_HERMES_EXE}}";
    private const string HermesExe = @"{{NAMED_HERMES_EXE}}";
    private const string HermesRoot = @"{{HERMES_ROOT}}";
    private const string HermesHome = @"{{HERMES_HOME}}";
    private const string UserData = @"{{USER_DATA}}";
    private const string WorkingDirectory = @"{{WORKING_DIRECTORY}}";
    private const string AppName = @"{{APP_NAME}}";
    private const string InstanceName = @"{{INSTANCE_NAME}}";
    private const string Aumid = @"{{AUMID}}";

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateHardLink(
        string newFileName,
        string existingFileName,
        IntPtr securityAttributes);

    [STAThread]
    private static int Main()
    {
        try
        {
            if (!File.Exists(SharedHermesExe))
                throw new FileNotFoundException("The shared Hermes Desktop executable was not found.", SharedHermesExe);
            if (!Directory.Exists(HermesRoot))
                throw new DirectoryNotFoundException("The shared Hermes runtime was not found: " + HermesRoot);

            EnsureSharedExecutableLink();
            Directory.CreateDirectory(HermesHome);
            Directory.CreateDirectory(UserData);

            var startInfo = new ProcessStartInfo
            {
                FileName = HermesExe,
                Arguments = "--user-data-dir=\"" + UserData + "\"",
                WorkingDirectory = WorkingDirectory,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            startInfo.EnvironmentVariables.Clear();
            startInfo.EnvironmentVariables["SystemRoot"] = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            startInfo.EnvironmentVariables["WINDIR"] = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            startInfo.EnvironmentVariables["USERPROFILE"] = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            startInfo.EnvironmentVariables["APPDATA"] = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            startInfo.EnvironmentVariables["LOCALAPPDATA"] = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            startInfo.EnvironmentVariables["TEMP"] = Path.GetTempPath();
            startInfo.EnvironmentVariables["TMP"] = Path.GetTempPath();
            startInfo.EnvironmentVariables["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? "";
            startInfo.EnvironmentVariables["HERMES_HOME"] = HermesHome;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_USER_DATA_DIR"] = UserData;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_HERMES_ROOT"] = HermesRoot;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_APP_NAME"] = AppName;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_CWD"] = WorkingDirectory;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_INSTANCE"] = InstanceName;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_AUMID"] = Aumid;
            startInfo.EnvironmentVariables["HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS"] = "1";
            startInfo.EnvironmentVariables["HERMES_DESKTOP_SKIP_PROTOCOL_REGISTER"] = "1";

            var process = Process.Start(startInfo);
            if (process == null)
                throw new InvalidOperationException("Windows did not create the " + AppName + " process.");

            WriteLaunchRecord(process.Id);
            return 0;
        }
        catch (Exception ex)
        {
            WriteErrorRecord(ex);
            MessageBox.Show(
                ex.Message,
                AppName + " launcher",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 1;
        }
    }

    private static void EnsureSharedExecutableLink()
    {
        // Rebuild the zero-copy hardlink on every clean launch so a local
        // Hermes update cannot leave the named executable on an old inode.
        // If the instance is already running, retain the in-use link and
        // let Electron focus that single-instance namespace.
        if (File.Exists(HermesExe))
        {
            try
            {
                File.Delete(HermesExe);
            }
            catch (IOException)
            {
                return;
            }
            catch (UnauthorizedAccessException)
            {
                return;
            }
        }

        if (!CreateHardLink(HermesExe, SharedHermesExe, IntPtr.Zero))
        {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "Windows could not create the shared " + AppName + " executable link.");
        }
    }

    private static void WriteLaunchRecord(int pid)
    {
        var text = new StringBuilder();
        text.AppendLine("{");
        text.AppendLine("  \"launched_at\": \"" + DateTime.UtcNow.ToString("o") + "\",");
        text.AppendLine("  \"pid\": " + pid + ",");
        text.AppendLine("  \"hermes_home\": \"" + JsonEscape(HermesHome) + "\",");
        text.AppendLine("  \"user_data\": \"" + JsonEscape(UserData) + "\",");
        text.AppendLine("  \"executable\": \"" + JsonEscape(HermesExe) + "\"");
        text.AppendLine("}");
        File.WriteAllText(Path.Combine(LauncherDirectory, "last-launch-native.json"), text.ToString(), Encoding.UTF8);
    }

    private static void WriteErrorRecord(Exception ex)
    {
        try
        {
            File.WriteAllText(
                Path.Combine(LauncherDirectory, "last-launch-native-error.txt"),
                DateTime.UtcNow.ToString("o") + Environment.NewLine + ex,
                Encoding.UTF8);
        }
        catch
        {
            // Preserve the original launch error for the message box.
        }
    }

    private static string JsonEscape(string value)
    {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
