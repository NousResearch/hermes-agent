using System;

internal static class HudModifierGestureTests
{
    private static void Check(bool expected, params uint[] modifiers)
    {
        HudModifierGesture gesture = new HudModifierGesture();
        bool summoned = false;
        for (int i = 0; i < modifiers.Length; i++)
            summoned |= gesture.Update(modifiers[i], false, false, i * 30);
        if (summoned != expected) throw new Exception("Unexpected gesture result");
    }

    private static void Reject(bool held, bool interrupted, long release)
    {
        HudModifierGesture gesture = new HudModifierGesture();
        gesture.Update(1, false, false, 0);
        gesture.Update(3, held, interrupted, 30);
        gesture.Update(2, false, false, 60);
        if (gesture.Update(0, false, false, release)) throw new Exception("Interrupted/long gesture summoned");
        // A rejected chord cannot poison the next clean tap.
        gesture.Update(2, false, false, 1000);
        gesture.Update(3, false, false, 1030);
        gesture.Update(1, false, false, 1060);
        if (!gesture.Update(0, false, false, 1090)) throw new Exception("Clean retry did not summon");
    }

    private static int Main()
    {
        Check(true, 1, 3, 2, 0);
        Check(true, 2, 3, 1, 0);
        Check(false, 1, 0);
        Check(false, 3, 1, 0); // Startup-held chord.
        Check(false, 1, 3, 1, 3, 2, 0); // Re-press during release.
        Check(false, 1, 7, 3, 1, 0); // AltGr, extra modifiers, both Ctrl keys.
        Reject(true, false, 90); // Another key or a held mouse button.
        Reject(false, true, 90); // Repeat, click, wheel, or duplicate transition.
        Reject(false, false, 501);
        Console.WriteLine("gesture contracts passed");
        return 0;
    }
}
