// Built by the C# compiler included with Windows' .NET Framework. Raw Input is
// passive: no hooks, event suppression, input injection, or recorded key data.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

internal sealed class HudModifierWindow : NativeWindow, IDisposable
{
    [StructLayout(LayoutKind.Sequential)]
    private struct RawDevice
    {
        internal ushort Page, Usage;
        internal uint Flags;
        internal IntPtr Target;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RawHeader
    {
        internal uint Type, Size;
        internal IntPtr Device, WParam;
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterRawInputDevices(RawDevice[] devices, uint count, uint size);
    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint GetRawInputData(IntPtr input, uint command, IntPtr data, ref uint size, uint headerSize);
    [DllImport("user32.dll")]
    private static extern short GetAsyncKeyState(int key);
    [DllImport("user32.dll")]
    internal static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);

    private readonly HudModifierGesture gesture = new HudModifierGesture();
    private readonly bool[] keys = new bool[256];
    private readonly Stopwatch clock = Stopwatch.StartNew();
    private readonly IntPtr buffer = Marshal.AllocHGlobal(64);
    private readonly int headerSize = Marshal.SizeOf(typeof(RawHeader));
    private uint mouseButtons;

    internal HudModifierWindow()
    {
        CreateHandle(new CreateParams { Caption = "Hermes HUD modifier input", Parent = new IntPtr(-3) });
        RawDevice[] devices = {
            new RawDevice { Page = 1, Usage = 6, Flags = 0x100 | 0x2000, Target = Handle },
            new RawDevice { Page = 1, Usage = 2, Flags = 0x100 | 0x2000, Target = Handle }
        };
        if (!RegisterRawInputDevices(devices, 2, (uint)Marshal.SizeOf(typeof(RawDevice))))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        Snapshot();
    }

    private static bool ModifierKey(int key)
    {
        return (key >= 0xA0 && key <= 0xA5) || key == 0x5B || key == 0x5C ||
            key == 0x10 || key == 0x11 || key == 0x12;
    }

    private uint Modifiers()
    {
        uint value = (keys[0xA2] || keys[0xA3]) ? 1u : 0u;
        if (keys[0xA4]) value |= 2;
        // Right Alt includes AltGr. It must never summon the HUD while typing.
        if (keys[0xA5] || keys[0xA0] || keys[0xA1] || keys[0x5B] || keys[0x5C] ||
            (keys[0xA2] && keys[0xA3])) value |= 4;
        return value;
    }

    private bool Held()
    {
        if (mouseButtons != 0) return true;
        for (int key = 8; key < 255; key++)
            if (keys[key] && !ModifierKey(key)) return true;
        return false;
    }

    private void Snapshot()
    {
        for (int key = 8; key < 255; key++) keys[key] = (GetAsyncKeyState(key) & 0x8000) != 0;
        int[] buttons = { 1, 2, 4, 5, 6 };
        mouseButtons = 0;
        for (int i = 0; i < buttons.Length; i++)
            if ((GetAsyncKeyState(buttons[i]) & 0x8000) != 0) mouseButtons |= 1u << i;
        gesture.Update(Modifiers(), Held(), true, clock.ElapsedMilliseconds);
    }

    private void ReadInput(IntPtr input)
    {
        uint size = 64;
        uint read = GetRawInputData(input, 0x10000003, buffer, ref size, (uint)headerSize);
        if (read == uint.MaxValue || read < headerSize)
        {
            HudModifierMonitor.Emit("{\"type\":\"error\",\"code\":\"unavailable\"}");
            Application.ExitThread();
            return;
        }
        uint type = (uint)Marshal.ReadInt32(buffer);
        IntPtr data = IntPtr.Add(buffer, headerSize);
        bool interrupted;
        if (type == 1 && read >= headerSize + 16)
        {
            int key = (ushort)Marshal.ReadInt16(data, 6);
            int flags = (ushort)Marshal.ReadInt16(data, 2);
            bool extended = (flags & 2) != 0;
            if (key == 0x11) key = extended ? 0xA3 : 0xA2;
            if (key == 0x12) key = extended ? 0xA5 : 0xA4;
            if (key == 0x10) key = Marshal.ReadInt16(data) == 0x36 ? 0xA1 : 0xA0;
            if (key < 8 || key >= 255) return;
            bool down = (flags & 1) == 0;
            interrupted = keys[key] == down || (key != 0xA2 && key != 0xA3 && key != 0xA4);
            keys[key] = down;
        }
        else if (type == 0 && read >= headerSize + 24)
        {
            uint flags = (ushort)Marshal.ReadInt16(data, 4);
            for (int i = 0; i < 5; i++)
            {
                if ((flags & (1u << (2 * i))) != 0) mouseButtons |= 1u << i;
                if ((flags & (2u << (2 * i))) != 0) mouseButtons &= ~(1u << i);
            }
            interrupted = flags != 0;
        }
        else return;
        if (gesture.Update(Modifiers(), Held(), interrupted, clock.ElapsedMilliseconds))
            HudModifierMonitor.Emit("{\"type\":\"summon\"}");
    }

    protected override void WndProc(ref Message message)
    {
        if (message.Msg == 0x00FF) ReadInput(message.LParam);
        else if (message.Msg == 0x00FE) Snapshot();
        else if (message.Msg == 0x0010)
        {
            Application.ExitThread();
            return;
        }
        base.WndProc(ref message);
    }

    public void Dispose()
    {
        DestroyHandle();
        Marshal.FreeHGlobal(buffer);
    }
}

internal static class HudModifierMonitor
{
    // Pipe writes never block the input loop, and a stalled parent cannot grow
    // an unbounded queue. Stdin EOF tears down Raw Input registration on exit.
    private static readonly BlockingCollection<string> Output = new BlockingCollection<string>(16);

    internal static void Emit(string message)
    {
        if (!Output.TryAdd(message)) Environment.Exit(3);
    }

    [STAThread]
    private static int Main(string[] args)
    {
        Thread writer = new Thread(delegate()
        {
            try
            {
                foreach (string message in Output.GetConsumingEnumerable())
                {
                    Console.Out.WriteLine(message);
                    Console.Out.Flush();
                }
            }
            catch (IOException) { Environment.Exit(3); }
        });
        writer.IsBackground = true;
        writer.Start();
        int exitCode = 0;
        try
        {
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.ThrowException);
            using (HudModifierWindow window = new HudModifierWindow())
            {
                Emit("{\"type\":\"ready\"}");
                if (Array.IndexOf(args, "--check") < 0)
                {
                    IntPtr handle = window.Handle;
                    Thread parent = new Thread(delegate()
                    {
                        try
                        {
                            using (Stream input = Console.OpenStandardInput())
                                while (input.ReadByte() >= 0) { }
                        }
                        catch (IOException) { }
                        HudModifierWindow.PostMessage(handle, 0x0010, IntPtr.Zero, IntPtr.Zero);
                    });
                    parent.IsBackground = true;
                    parent.Start();
                    Application.Run();
                }
            }
        }
        catch (Exception)
        {
            Emit("{\"type\":\"error\",\"code\":\"unavailable\"}");
            exitCode = 1;
        }
        Output.CompleteAdding();
        return writer.Join(1000) ? exitCode : 3;
    }
}
