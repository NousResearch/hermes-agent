"""Object-native motion.

The failure this fixes: told to animate something, a video model defaults to
moving the whole object — bouncing, wobbling, drifting, jittering. It passes
every motion metric and still looks wrong, because almost nothing in the world
moves by bouncing. It reads as a sticker being shaken rather than the thing
itself being alive.

A flame licks upward and its tips tear away. A clock's hand sweeps at a
constant rate. A droplet swells, hangs, and falls. That motion is specific to
the object and cannot be guessed from the word "animate".

So each preset names two things:
  physics — what this object actually does, in physical language
  anchor  — what must NOT move, which is usually the whole silhouette

The quality words (overshoot, settle, morph) are borrowed from Emil Kowalski's
animation vocabulary. They describe how a motion should feel, which is worth
keeping — they just cannot substitute for knowing what the object does.
"""

# Applied to every preset. These are the defaults a model reaches for, and they
# are wrong for almost every icon.
# An earlier version of this file banned all whole-object motion outright. It
# was written to stop two real failures — a turntable spin, and a slow aimless
# bob — and it stopped them by freezing everything. Measured on one run: the
# object's centre moved 0.3px in 384 and its area changed 0.2%. Rigid, lifeless
# and technically passing every check.
#
# The mistake was treating body motion as the problem. It never was. The
# problem is body motion with no purpose: continuous, ambient, going nowhere.
# Purposeful body motion — a squash, a recoil, a shake, an anticipation before
# an action — is most of what makes animation feel alive, and forbidding it
# throws away the entire vocabulary to avoid two specific abuses.
#
# So the body gets an energy budget instead of a ban, and only the two actual
# abuses stay banned.
ENERGY = {
    "still": (
        "The object itself holds completely still. Its outline, position and "
        "facing do not change at all"
    ),
    "calm": (
        "The object may shift a little as it acts — a slight lean, a settle, a "
        "gentle breath of scale — staying within a few percent of its size and "
        "always coming back to rest"
    ),
    "lively": (
        "The object moves as part of what it is doing: it squashes and "
        "stretches, tilts, recoils, shakes briefly, dips and springs back. "
        "These movements are fast and decisive rather than constant, are a "
        "modest fraction of its size, and each one resolves back to its resting "
        "pose"
    ),
    "playful": (
        "The object behaves like a character. It anticipates before it acts, "
        "squashes and stretches generously, overshoots and wobbles to a stop, "
        "and may hop or shake in place. The exaggeration is deliberate and can "
        "be large, as long as it stays centred, stays well inside the frame, "
        "and returns exactly to its resting pose"
    ),
}

# Only the two abuses, stated precisely enough that energy cannot be read as
# permission for them.
NEVER = (
    "It never rotates, turns, spins or orbits as a whole — no turntable, no "
    "revolve, no tumbling — and its facing stays the same throughout. It never "
    "drifts, floats or wanders: it is centred and at rest at both the first and "
    "the last frame, and any displacement in between belongs to a brief "
    "deliberate action, never to a constant ambient motion."
)

# Objects differ in whether they move at all, and that decides the whole shape
# of the animation. Naming presets one object at a time does not scale and
# never covers the next request, so pick a STRATEGY instead: four of them
# cover essentially everything, and the choice follows from one question —
# what, if anything, does this thing do when left alone?
#
# Getting this wrong is what produces the two classic failures. Ask an inert
# object to "move naturally" and you get a turntable spin or a gentle bob,
# because it has no natural motion and the model falls back on the generic.
# Ask a living or flowing thing to perform a discrete event and you get a
# stiff, mechanical beat where continuous motion belonged.
STRATEGIES = {
    "native": (
        "This object moves of its own accord, so animate what it physically "
        "does: the forces acting on it, the way its material behaves, the "
        "rhythm it naturally has. The motion is continuous and never resolves "
        "into a single mechanical cycle"
    ),
    "event": (
        "This object does not move on its own, so it performs its function "
        "once: a single clear action with a beginning, a peak and a return to "
        "rest, followed by a pause before it happens again. The action is fast "
        "and decisive and occupies only a short part of the clip — the rest is "
        "the object at rest, which is what makes the action read"
    ),
    "part": (
        "The body of the object is an anchor: its outline and its position are "
        "fixed for the entire clip. Only one small part of it moves — the part "
        "that is loose, hinged, hanging or light. The moving part is a small "
        "fraction of the whole, and it moves under its own weight or its own "
        "tension, never because the body carries it"
    ),
    "surface": (
        "The object's outline does not change and the object itself does not "
        "move. Everything happens on or just under its surface: light, a "
        "highlight or a sheen travelling across it; a colour or material state "
        "spreading through it; and small local features forming, swelling and "
        "resolving in place — the surface is alive while the shape is still"
    ),
}

# The subtlest way the whole-object ban gets defeated: the model looks for a
# CAUSE for the motion, and the most obvious cause is usually the object being
# moved by something. It then animates the cause rather than the effect, and
# the result is a displaced object with a perfectly good excuse — which still
# reads as the icon bobbing, because at icon size the excuse is invisible and
# only the displacement is legible.
#
# So: the driving force acts on the object where it stands.
CAUSE = (
    "Any movement the object makes is its own — its action, its recoil, its "
    "weight, or a force already in the scene such as gravity or heat. Nothing "
    "outside the frame moves it: no hand, no prop, no unseen mount, and it is "
    "never carried, dragged or repositioned by something else. If an effect "
    "would normally come from the object being handled, show the effect and let "
    "the object produce it itself."
)
# Two general moves that make almost any icon read as alive, and that no
# strategy gets to skip.
#
# The first is scale. A single large motion of the whole object is the obvious
# thing to reach for and the worst of the options: it looks generic and carries
# no information about what the object is. Many small motions distributed over
# the object's own details cost nothing and are specific to that object by
# construction.
#
# The second is phase. Anything an object has several of will, left alone, be
# animated in unison, and unison is the single loudest tell that something was
# animated rather than observed. Real repeated things drift out of step.
DETAIL = (
    "Favour many small motions in the object's own details over one large "
    "motion of the whole. Small features may form, swell, travel a short "
    "distance and resolve in place. Where the object has several of the same "
    "element, each one moves independently and slightly out of step with the "
    "others — never together, never in a single synchronised beat."
)

# Containment stated as physics rather than as a boundary. A boundary is a
# constraint the model can violate without noticing; a round trip is a property
# of the motion itself, and it is also exactly what a seamless loop needs.
RETURN = (
    "Anything that extends, rises, spreads or is thrown off also comes back: "
    "it retracts, settles, is reabsorbed, or fades out entirely. No part of the "
    "motion travels in one direction and keeps going."
)

# Drama and magnitude are not the same thing, and conflating them is how an
# energetic instruction turns into a violent one. What makes a small action
# read is the contrast with the stillness around it — the pause before, the
# speed of the move, the settle after. Magnitude adds nothing an icon can use:
# at the size these are actually viewed, a large movement is just a blur, and
# anything thrown hard simply leaves.
RESTRAINT = (
    "Keep the whole thing understated. This is a small icon, so the motion "
    "reads through its timing — a held pause, a quick move, a settle — and not "
    "through how far anything travels. Every movement stays modest in size and "
    "close to where it began; nothing is violent, forceful or explosive, and "
    "nothing flies, splashes or scatters."
)

# A high energy budget is an invitation to invent, and what gets invented is
# usually an event the object could not actually perform: a sealed thing
# releasing its contents, a solid thing behaving like a liquid, something
# appearing that the object has no way to produce. It reads as wrong instantly
# even when the motion itself is well made.
#
# Energy should buy exaggeration of timing and deformation, never a new event.
PLAUSIBLE = (
    "Everything that happens must be something this object could plausibly do "
    "in its ordinary use, in exactly the state it is shown in. Do not invent "
    "events it could not perform: what is closed or sealed stays closed unless "
    "opening it is the action itself, nothing escapes a container that is not "
    "open, and nothing appears that this object has no way to produce. "
    "Exaggeration belongs to the timing and the deformation, never to adding "
    "an event that could not happen."
)

# Permission, not instruction. Emitted elements are the difference between an
# inert object being interesting and being furniture, but they are also the
# thing a video model is most eager to overdo, so this stays opt-in and the
# constraints on it are tight.
EMIT = (
    "The object may briefly produce small elements of its own — a fragment, a "
    "droplet, a spark, a puff, a glint. Keep these few and tiny — a small "
    "number of them, each a very small fraction of the object, consistent with "
    "what the object is made of or holds. They drift only a short distance from "
    "where they appeared, stay well inside the frame, and fade out or are "
    "reabsorbed within a few frames. Never a spray, burst, splash or shower, "
    "and nothing is ever thrown hard or far. Nothing ever enters from outside"
)

# What the object may also do is TEMPORARILY LOSE PART OF ITSELF and get it
# back before the loop closes — a piece removed, a section opened, a state
# changed and restored. This is worth saying explicitly because "keep the same
# identity" otherwise reads as "never change", and an object that can never be
# altered can only ever bob.
TRANSIENT = (
    "A change to the object itself is allowed during the action — part of it "
    "may be removed, opened, split, filled or emptied — provided it returns to "
    "exactly its opening state by the end so the loop closes seamlessly"
)

ARCHETYPES = {
    "flame": dict(
        physics="fire rises: the flame licks upward, its tips tapering, curling "
                "and tearing away into nothing while new flame feeds up from the "
                "base; the inner core swells and sinks in its own slower rhythm",
        anchor="the base of the flame stays planted and does not move",
    ),
    "clock": dict(
        physics="the hand sweeps steadily clockwise at a constant rate, all the "
                "way around, exactly as a real clock hand moves; the case and "
                "dial stay perfectly still",
        anchor="only the hand moves — the body, face and markings are fixed",
    ),
    "droplet": dict(
        physics="the surface tension works: the drop swells at the bottom, "
                "stretches, hangs, and its highlight slides as the surface moves, "
                "the way real water skins and settles",
        anchor="the drop stays in place and keeps its teardrop identity",
    ),
    "heart": dict(
        physics="it beats: a quick double pulse, swelling then settling back, "
                "the way a heartbeat has two knocks and a rest between",
        anchor="it beats in place, centred, and never travels",
    ),
    "bell": dict(
        physics="the bell rings: the body tilts and the clapper swings inside it, "
                "the swing decaying naturally, then rest before it rings again",
        anchor="the bell pivots from its crown, which stays fixed",
    ),
    "star": dict(
        physics="it twinkles: the points brighten and lengthen in turn, a "
                "specular glint travelling across the surface",
        anchor="the star's centre and outline stay put",
    ),
    "leaf": dict(
        physics="it responds to a breeze: the blade flexes and twists along its "
                "spine, edges lifting and settling, the stem holding it back",
        anchor="the stem is fixed; only the blade flexes",
    ),
    "gear": dict(
        physics="it rotates steadily about its own centre at a constant rate, "
                "teeth passing evenly, exactly as a driven gear turns",
        anchor="the centre is fixed and the rotation never changes speed",
    ),
    "battery": dict(
        physics="it charges: the fill level climbs the cell and the charge "
                "indicator pulses as it rises, then resets",
        anchor="the casing is rigid and completely still",
    ),
    "camera": dict(
        physics="it takes a photo: the shutter button presses down and springs "
                "back, the lens iris blinks shut and open, and the flash fires a "
                "bright burst of light out of the lens with small sparkles "
                "flicking outward and fading; then it rests before firing again",
        anchor="the camera body is rigid and completely still — only the button, "
               "the iris and the light change",
        accent=True,
    ),
    "cloud": dict(
        physics="it churns: the puffs roll and fold into one another slowly, the "
                "silhouette breathing in and out at its edges",
        anchor="the cloud stays centred and keeps its overall mass",
    ),
}

QUALITY = {
    "settle": "movements overshoot slightly and settle rather than stopping dead",
    "snappy": "each movement is fast and decisive, with a held beat between",
    "smooth": "the motion is even and unhurried throughout",
    "organic": "no two cycles are identical; the rhythm varies slightly",
}


DEFAULT_ENERGY = {"native": "lively", "event": "lively",
                  "part": "calm", "surface": "still"}


def compose(motion=None, preset=None, quality=None, strategy=None, emit=False,
            energy=None):
    """Build the motion clause: strategy, what it does, what stays put, feel."""
    parts = []
    if strategy:
        parts.append(STRATEGIES[strategy])
        if strategy in ("event", "part"):
            parts.append(TRANSIENT)
    if emit:
        parts.append(EMIT)
    if preset:
        a = ARCHETYPES[preset]
        parts.append(a["physics"])
        parts.append(a["anchor"])
        if a.get("accent"):
            parts.append(ACCENT)
    if motion:
        parts.append(motion.strip().rstrip("."))
    if quality:
        parts.append(QUALITY[quality])
    if not parts:
        raise SystemExit("Give --strategy, --motion or --preset.")
    parts.append(ENERGY[energy or DEFAULT_ENERGY.get(strategy, "lively")].rstrip("."))
    parts.append(RESTRAINT.rstrip("."))
    parts.append(PLAUSIBLE.rstrip("."))
    parts.append(DETAIL.rstrip("."))
    parts.append(RETURN.rstrip("."))
    parts.append(NEVER.rstrip("."))
    parts.append(CAUSE.rstrip("."))
    return ". ".join(p.strip().rstrip(".") for p in parts) + "."


def describe():
    return "\n".join(f"  {k:9s} {v['physics'][:66]}..." for k, v in ARCHETYPES.items())
