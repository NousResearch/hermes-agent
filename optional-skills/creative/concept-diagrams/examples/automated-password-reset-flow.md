# Automated Password Reset Flow

A two-section flowchart tracing the full user journey for a web application password reset: the initial request phase (forgot password → email check → token generation) and the reset-form phase (link click → new password entry → token/password validation). Demonstrates multi-exit decision diamonds, a three-column branching layout, a loop-back path, and a cross-section separator arrow.

## Key Patterns Used

- **Three-column layout**: Left column (error/terminal branches at cx=115), center column (main happy path at cx=340), right column (expired-token branch at cx=552) — allows side branches to live at the same y-level as center nodes without overlap
- **Decision diamonds with `<polygon>`**: Each decision uses a `<g class="decision">` wrapper containing a `<polygon>` and centered `<text>`; the diamond points are computed as `cx±hw, cy±hh` (hw=100, hh=28)
- **Pill-shaped terminals**: Start and end nodes use `rx=22` on their `<rect>` to signal entry/exit points; all mid-flow process nodes use `rx=8`
- **Three-branch decision paths**: Each diamond has a "Yes" branch (down, short `<line>`) and a "No" branch (`<path>` going horizontal then vertical to a side column)
- **Loop-back path**: Mismatch error node loops back to the password-entry node via a routing corridor at x=215 — a 5-px gap between the left column (right edge x=210) and center column (left edge x=220); the path exits the bottom of the error node, drops below it, travels right to x=215, then goes up to the target node's center y, then right 5 px into the node's left edge
- **Section separator**: A dashed horizontal `<line>` at y=452 splits the two phases; the connecting arrow crosses it with a faded label ("user receives email") to preserve flow continuity
- **Italic annotation**: The exact UX copy for the generic message ("If that email exists…") is shown as a faded italic `ts` text block below the left-branch terminal node
- **Legend row**: Five inline swatches (gray, purple, teal, red, amber diamond) at the bottom explain the color-to-role mapping

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 960" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!--
    Column layout (680px viewBox, safe area x=40–640):
      Left  col : x=20,  w=190, cx=115  (error / terminal branches)
      Center col: x=220, w=240, cx=340  (main happy path)
      Right  col: x=465, w=175, cx=552  (expired-token branch)
      Loop corridor at x=215 (5-px gap between left and center cols)
  -->

  <!-- ═══ SECTION 1 — Forgot password request ═══ -->
  <text class="ts" x="40" y="38" opacity=".45">Section 1 — Forgot password request</text>

  <!-- START terminal (pill rx=22 signals start/end) -->
  <g class="c-gray">
    <rect x="220" y="46" width="240" height="44" rx="22"/>
    <text class="th" x="340" y="68" text-anchor="middle" dominant-baseline="central">User: &quot;Forgot password&quot;</text>
  </g>

  <line x1="340" y1="90" x2="340" y2="108" class="arr" marker-end="url(#arrow)"/>

  <!-- N2 · Enter email -->
  <g class="c-gray">
    <rect x="220" y="108" width="240" height="44" rx="8"/>
    <text class="th" x="340" y="130" text-anchor="middle" dominant-baseline="central">Enter email address</text>
  </g>

  <line x1="340" y1="152" x2="340" y2="172" class="arr" marker-end="url(#arrow)"/>

  <!-- D1 · Email in system?  diamond: center=(340,200) hw=100 hh=28 -->
  <g class="decision">
    <polygon points="340,172 440,200 340,228 240,200"/>
    <text class="th" x="340" y="200" text-anchor="middle" dominant-baseline="central">Email in system?</text>
  </g>

  <!-- D1 "No" → left column -->
  <path d="M 240,200 L 115,200 L 115,248" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="178" y="193" text-anchor="middle" opacity=".75">No</text>

  <!-- D1 "Yes" → continue down -->
  <line x1="340" y1="228" x2="340" y2="248" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="348" y="242" text-anchor="start" opacity=".75">Yes</text>

  <!-- ── Left branch (D1 = No): generic security message → end ── -->

  <!-- L1 · Generic message (security: never confirm email existence) -->
  <g class="c-gray">
    <rect x="20" y="248" width="190" height="56" rx="8"/>
    <text class="th" x="115" y="269" text-anchor="middle" dominant-baseline="central">Generic message shown</text>
    <text class="ts" x="115" y="287" text-anchor="middle" dominant-baseline="central">Email sent if found</text>
  </g>

  <line x1="115" y1="304" x2="115" y2="324" class="arr" marker-end="url(#arrow)"/>

  <!-- L2 · End terminal (left) -->
  <g class="c-gray">
    <rect x="20" y="324" width="190" height="44" rx="22"/>
    <text class="th" x="115" y="346" text-anchor="middle" dominant-baseline="central">Request handled</text>
  </g>

  <!-- Italic annotation: actual UX copy shown below the end node -->
  <text class="ts" x="20" y="384" opacity=".45" font-style="italic">&quot;If that email exists, a reset</text>
  <text class="ts" x="20" y="398" opacity=".45" font-style="italic">link has been sent.&quot;</text>

  <!-- ── Center Yes branch: system generates & sends token ── -->

  <!-- N3 · Generate unique token -->
  <g class="c-purple">
    <rect x="220" y="248" width="240" height="56" rx="8"/>
    <text class="th" x="340" y="269" text-anchor="middle" dominant-baseline="central">Generate unique token</text>
    <text class="ts" x="340" y="287" text-anchor="middle" dominant-baseline="central">Time-limited, cryptographic</text>
  </g>

  <line x1="340" y1="304" x2="340" y2="324" class="arr" marker-end="url(#arrow)"/>

  <!-- N4 · Store token + user ID -->
  <g class="c-purple">
    <rect x="220" y="324" width="240" height="44" rx="8"/>
    <text class="th" x="340" y="346" text-anchor="middle" dominant-baseline="central">Store token + user ID</text>
  </g>

  <line x1="340" y1="368" x2="340" y2="388" class="arr" marker-end="url(#arrow)"/>

  <!-- N5 · Send reset email -->
  <g class="c-teal">
    <rect x="220" y="388" width="240" height="44" rx="8"/>
    <text class="th" x="340" y="410" text-anchor="middle" dominant-baseline="central">Send reset link via email</text>
  </g>

  <!-- ═══ Section separator ═══ -->
  <line x1="40" y1="452" x2="640" y2="452"
        stroke="var(--border)" stroke-width="1" stroke-dasharray="8 5"/>

  <!-- Arrow crossing separator (with inline label) -->
  <line x1="340" y1="432" x2="340" y2="472" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="348" y="448" text-anchor="start" opacity=".55">user receives email</text>

  <text class="ts" x="40" y="464" opacity=".45">Section 2 — Password reset form</text>

  <!-- ═══ SECTION 2 — Password reset form ═══ -->

  <!-- N6 · User clicks reset link -->
  <g class="c-gray">
    <rect x="220" y="480" width="240" height="44" rx="8"/>
    <text class="th" x="340" y="502" text-anchor="middle" dominant-baseline="central">User clicks reset link</text>
  </g>

  <line x1="340" y1="524" x2="340" y2="544" class="arr" marker-end="url(#arrow)"/>

  <!-- N7 · Enter new password ×2 -->
  <g class="c-gray">
    <rect x="220" y="544" width="240" height="56" rx="8"/>
    <text class="th" x="340" y="565" text-anchor="middle" dominant-baseline="central">Enter new password ×2</text>
    <text class="ts" x="340" y="583" text-anchor="middle" dominant-baseline="central">Confirm both passwords match</text>
  </g>

  <line x1="340" y1="600" x2="340" y2="620" class="arr" marker-end="url(#arrow)"/>

  <!-- D2 · Token expired?  diamond: center=(340,648) hw=100 hh=28 -->
  <g class="decision">
    <polygon points="340,620 440,648 340,676 240,648"/>
    <text class="th" x="340" y="648" text-anchor="middle" dominant-baseline="central">Token expired?</text>
  </g>

  <!-- D2 "Yes" → right column (expired-token branch) -->
  <path d="M 440,648 L 552,648 L 552,692" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="496" y="641" text-anchor="middle" opacity=".75">Yes</text>

  <!-- D2 "No" → down to password-match check -->
  <line x1="340" y1="676" x2="340" y2="714" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="348" y="698" text-anchor="start" opacity=".75">No</text>

  <!-- ── Right branch (D2 = Yes): token expired → dead end ── -->

  <!-- R1 · Token expired error -->
  <g class="c-red">
    <rect x="465" y="692" width="175" height="56" rx="8"/>
    <text class="th" x="552" y="713" text-anchor="middle" dominant-baseline="central">Token expired</text>
    <text class="ts" x="552" y="731" text-anchor="middle" dominant-baseline="central">Show expiry error</text>
  </g>

  <line x1="552" y1="748" x2="552" y2="768" class="arr" marker-end="url(#arrow)"/>

  <!-- R2 · End terminal (right) -->
  <g class="c-gray">
    <rect x="465" y="768" width="175" height="44" rx="22"/>
    <text class="th" x="552" y="790" text-anchor="middle" dominant-baseline="central">End — request again</text>
  </g>

  <!-- D3 · Passwords match?  diamond: center=(340,742) hw=100 hh=28 -->
  <g class="decision">
    <polygon points="340,714 440,742 340,770 240,742"/>
    <text class="th" x="340" y="742" text-anchor="middle" dominant-baseline="central">Passwords match?</text>
  </g>

  <!-- D3 "No" → left column (mismatch branch) -->
  <path d="M 240,742 L 115,742 L 115,786" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="178" y="735" text-anchor="middle" opacity=".75">No</text>

  <!-- D3 "Yes" → down to reset -->
  <line x1="340" y1="770" x2="340" y2="790" class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="348" y="783" text-anchor="start" opacity=".75">Yes</text>

  <!-- ── Left branch (D3 = No): passwords don't match → loop back ── -->

  <!-- L3 · Password mismatch error -->
  <g class="c-red">
    <rect x="20" y="786" width="190" height="56" rx="8"/>
    <text class="th" x="115" y="807" text-anchor="middle" dominant-baseline="central">Password mismatch</text>
    <text class="ts" x="115" y="825" text-anchor="middle" dominant-baseline="central">Passwords do not match</text>
  </g>

  <!-- Loop-back arrow: exits L3 bottom → drops to y=862 →
       travels right to corridor x=215 → climbs to N7 center y=572 →
       enters N7 left edge at (220, 572) pointing right -->
  <path d="M 115,842 L 115,862 L 215,862 L 215,572 L 220,572"
        class="arr" marker-end="url(#arrow)"/>
  <text class="ts" x="224" y="538" text-anchor="start" opacity=".6">retry</text>

  <!-- ── Center Yes branch (D3 = Yes): reset password & invalidate token ── -->

  <!-- N8 · Reset password -->
  <g class="c-teal">
    <rect x="220" y="790" width="240" height="56" rx="8"/>
    <text class="th" x="340" y="811" text-anchor="middle" dominant-baseline="central">Reset password</text>
    <text class="ts" x="340" y="829" text-anchor="middle" dominant-baseline="central">Invalidate used token</text>
  </g>

  <line x1="340" y1="846" x2="340" y2="866" class="arr" marker-end="url(#arrow)"/>

  <!-- N9 · Success terminal -->
  <g class="c-green">
    <rect x="220" y="866" width="240" height="44" rx="22"/>
    <text class="th" x="340" y="888" text-anchor="middle" dominant-baseline="central">Password reset complete</text>
  </g>

  <!-- ═══ Legend ═══ -->
  <text class="ts" x="40" y="930" opacity=".4">Legend —</text>
  <rect x="108" y="920" width="13" height="13" rx="2" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text class="ts" x="126" y="930" opacity=".7">User action</text>
  <rect x="210" y="920" width="13" height="13" rx="2" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text class="ts" x="228" y="930" opacity=".7">System process</text>
  <rect x="334" y="920" width="13" height="13" rx="2" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text class="ts" x="352" y="930" opacity=".7">Email / success</text>
  <rect x="455" y="920" width="13" height="13" rx="2" fill="#FCEBEB" stroke="#A32D2D" stroke-width="0.5"/>
  <text class="ts" x="473" y="930" opacity=".7">Error state</text>
  <polygon points="556,926 566,932 556,938 546,932" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
  <text class="ts" x="572" y="932" opacity=".7">Decision</text>

</svg>
```

## Custom CSS

Add these classes to the hosting page `<style>` block (in addition to the standard skill CSS):

```css
/* Decision diamond — amber fill, same palette as c-amber */
.decision > polygon { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.5; }
.decision > .th     { fill: #633806; }

@media (prefers-color-scheme: dark) {
  .decision > polygon { fill: #633806; stroke: #EF9F27; }
  .decision > .th     { fill: #FAC775; }
}
```

## Color Assignments

| Element | Color | Reason |
|---------|-------|--------|
| Start / end terminals | `c-gray` | Neutral entry and exit points |
| User actions (enter email, click link, enter password) | `c-gray` | User-facing steps with no system processing |
| Generic message + request-handled terminal | `c-gray` | Intentionally neutral — the security message must not reveal data |
| Generate & store token | `c-purple` | Backend system operations |
| Send reset email | `c-teal` | Positive external action (outbound communication) |
| Token expired error | `c-red` | Failure / blocking error state |
| Password mismatch error | `c-red` | Validation failure |
| Reset password + success | `c-teal` / `c-green` | Positive outcome: teal for the action, green pill for the terminal |
| Decision diamonds | `c-amber` (custom `.decision`) | Warning / branch point — matches amber semantic meaning |

## Layout Notes

- **ViewBox**: 680×960 — tall flowchart with two phases
- **Three-column structure**: Left (cx=115), center (cx=340), right (cx=552) — each branch stays within its column; only `<path>` arrows cross column boundaries
- **Diamond formula**: `<polygon points="cx,cy-hh cx+hw,cy cx,cy+hh cx-hw,cy"/>` with hw=100, hh=28 gives a 200×56px diamond that sits flush with the center column (x=220–460)
- **Branch routing pattern**: "No" paths use `<path d="M left_point,cy L side_cx,cy L side_cx,node_top">` — one horizontal segment + one vertical segment, no curves needed
- **Loop corridor**: The 5-px gap at x=210–220 between left and center columns provides a clean vertical channel for the loop-back path without any node overlap; the path exits node bottom, drops 20px, goes right to x=215, climbs to target y, enters from left
- **Section separator**: A dashed `<line>` at y=452 with `stroke-dasharray="8 5"` provides a visual phase break; the single connecting arrow crosses it at center, with a faded label on the arrow
- **Pill terminals**: `rx=22` (half the 44px node height) produces a perfect capsule/pill shape — use this consistently for all start/end terminals
- **Error annotation**: The exact UX copy is rendered as faded (`opacity=".45"`) italic `ts` text below the relevant node, keeping it informative without cluttering the flow
