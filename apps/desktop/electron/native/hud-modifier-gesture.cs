// The same bounded, release-only gesture contract as hud-modifier-gesture.h.
// No key history is retained, only the current chord's summary.
internal sealed class HudModifierGesture
{
    private uint modifiers;
    private bool started, both, blocked, releasing;
    private long began;

    internal bool Update(uint next, bool held, bool interrupted, long now)
    {
        if (held || interrupted || (next & ~3u) != 0) blocked = true;
        uint pressed = next & ~modifiers;
        if (releasing && pressed != 0) blocked = true;
        if ((modifiers & ~next) != 0) releasing = true;
        if (modifiers == 0 && !blocked && (next == 1 || next == 2))
        {
            started = true;
            began = now;
        }
        if (next != 0 && !started) blocked = true;
        if (next == 3 && started) both = true;
        bool summon = next == 0 && both && !blocked && now >= began && now - began <= 500;
        modifiers = next;
        if (next == 0 && !held)
        {
            started = both = blocked = releasing = false;
            began = 0;
        }
        return summon;
    }
}
