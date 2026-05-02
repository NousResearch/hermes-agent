# Place Order — UML Sequence Diagram

A UML sequence diagram for the 'Place Order' use case in an e-commerce system. Six lifelines (:Customer, :ShoppingCart, :OrderController, :PaymentGateway, :InventorySystem, :EmailService) interact across 14 numbered messages. An **alt** combined fragment (amber) covers the three conditional outcomes — payment authorized, payment failed, and item unavailable. A **par** combined fragment (teal) nested inside the success branch shows concurrent email confirmation and stock-level update. Demonstrates activation bars, two distinct arrowhead types, UML pentagon fragment tags, and guard conditions.

## Key Patterns Used

- **6 lifelines at equal spacing**: Lifeline centers placed at x=90, 190, 290, 390, 490, 590 (100px apart) so the first box left-edge lands at x=40 and the last right-edge lands at x=640 — exactly filling the safe area
- **Two-row actor headers**: Each lifeline box shows `":"` (small, tertiary color) on one line and the class name (slightly larger, bold) on a second line, matching the UML anonymous-instance notation `:ClassName`
- **Two separate arrowhead markers**: `#arr-call` is a filled triangle (`<polygon>`) for synchronous calls; `#arr-ret` is an open chevron (`fill="none"`) for dashed return messages — both use `context-stroke` to inherit line color
- **Activation bars**: Narrow 8px-wide rectangles (`class="activation"`) layered on top of lifeline stems to show object execution periods; OrderController's bar spans the entire interaction; shorter bars mark PaymentGateway, InventorySystem, and EmailService during their active windows
- **Combined fragment pentagon tag**: Each `alt` / `par` frame uses a `<polygon>` dog-eared label shape in the top-left corner — points follow the pattern `(x,y) (x+w,y) (x+w+6,y+6) (x+w+6,y+18) (x,y+18)` creating the characteristic UML notch
- **Nested par inside alt**: The `par` rect (teal) sits inside branch 1 of the `alt` rect (amber); inner rect uses inset x/y (+15/+2) so both borders remain visible and distinguishable
- **Guard conditions**: Italic text in `[square brackets]` placed immediately after each alt frame divider line, or just inside the top frame for branch 1 — rendered with a dedicated `guard-lbl` class (italic, amber color)
- **Alt branch dividers**: Solid horizontal lines (`.frag-alt-div`) span the full alt rect width to separate the three branches; par branch separator uses a dashed line (`.frag-par-div`) per UML spec
- **Lifeline end caps**: Short 14px horizontal tick marks at y=590 (bottom of all lifeline stems) to formally terminate each lifeline
- **Message sequence annotation**: A faint counter row below the legend (①–③ / ④–⑩ / ⑪–⑫ / ⑬–⑭) explains the four message groups without adding noise to the diagram body

## Diagram

```xml
<svg width="100%" viewBox="0 0 680 648" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Open chevron arrowhead — return messages -->
    <marker id="arr-ret" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>

    <!-- Filled triangle arrowhead — synchronous calls -->
    <marker id="arr-call" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <polygon points="0,1 10,5 0,9" fill="context-stroke"/>
    </marker>
  </defs>

  <!--
    Lifeline centres (x):
      L1 :Customer        →  90
      L2 :ShoppingCart    → 190
      L3 :OrderController → 290
      L4 :PaymentGateway  → 390
      L5 :InventorySystem → 490
      L6 :EmailService    → 590
    Actor boxes: x = cx−50, y=20, w=100, h=56, rx=6
    Lifelines:   x = cx,    y1=76, y2=590
  -->

  <!-- ── 1. LIFELINE DASHED STEMS (drawn first, behind everything) ── -->
  <line x1="90"  y1="76" x2="90"  y2="590" class="lifeline"/>
  <line x1="190" y1="76" x2="190" y2="590" class="lifeline"/>
  <line x1="290" y1="76" x2="290" y2="590" class="lifeline"/>
  <line x1="390" y1="76" x2="390" y2="590" class="lifeline"/>
  <line x1="490" y1="76" x2="490" y2="590" class="lifeline"/>
  <line x1="590" y1="76" x2="590" y2="590" class="lifeline"/>

  <!-- ── 2. ACTOR HEADER BOXES ── -->

  <!-- :Customer -->
  <rect x="40"  y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="90"  y="40" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="90"  y="58" text-anchor="middle" dominant-baseline="central">Customer</text>

  <!-- :ShoppingCart -->
  <rect x="140" y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="190" y="37" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="190" y="55" text-anchor="middle" dominant-baseline="central">ShoppingCart</text>

  <!-- :OrderController -->
  <rect x="240" y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="290" y="37" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="290" y="55" text-anchor="middle" dominant-baseline="central">OrderController</text>

  <!-- :PaymentGateway -->
  <rect x="340" y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="390" y="37" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="390" y="55" text-anchor="middle" dominant-baseline="central">PaymentGateway</text>

  <!-- :InventorySystem -->
  <rect x="440" y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="490" y="37" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="490" y="55" text-anchor="middle" dominant-baseline="central">InventorySystem</text>

  <!-- :EmailService -->
  <rect x="540" y="20" width="100" height="56" rx="6" class="actor"/>
  <text class="actor-colon" x="590" y="37" text-anchor="middle" dominant-baseline="central">:</text>
  <text class="actor-name"  x="590" y="55" text-anchor="middle" dominant-baseline="central">EmailService</text>

  <!-- ── 3. ACTIVATION BARS ── -->
  <!-- ShoppingCart: active while forwarding checkout → placeOrder -->
  <rect x="186" y="102" width="8" height="26"  rx="1" class="activation"/>
  <!-- OrderController: active throughout full sequence -->
  <rect x="286" y="128" width="8" height="415" rx="1" class="activation"/>
  <!-- PaymentGateway: active during auth check (happy-path branch only) -->
  <rect x="386" y="154" width="8" height="46"  rx="1" class="activation"/>
  <!-- InventorySystem: active from reserveItems → updateStockLevels end -->
  <rect x="486" y="225" width="8" height="128" rx="1" class="activation"/>
  <!-- EmailService: active during confirmation send -->
  <rect x="586" y="290" width="8" height="25"  rx="1" class="activation"/>

  <!-- ── 4. PRE-ALT MESSAGES ── -->

  <!-- ① checkout()  :Customer → :ShoppingCart -->
  <line x1="90"  y1="102" x2="186" y2="102" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="140" y="97" text-anchor="middle">checkout()</text>

  <!-- ② placeOrder(cartItems)  :ShoppingCart → :OrderController -->
  <line x1="194" y1="128" x2="286" y2="128" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="242" y="123" text-anchor="middle">placeOrder(cartItems)</text>

  <!-- ③ authorizePayment(amount)  :OrderController → :PaymentGateway -->
  <line x1="294" y1="154" x2="386" y2="154" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="342" y="149" text-anchor="middle">authorizePayment(amount)</text>

  <!-- ── 5. ALT COMBINED FRAGMENT  y=166 → y=563 ── -->

  <!-- Outer alt rectangle -->
  <rect x="45" y="166" width="590" height="397" rx="3" class="frag-alt-bg"/>

  <!-- Pentagon "alt" tag: TL corner notch shape -->
  <polygon points="45,166 84,166 90,173 90,185 45,185" class="frag-alt-tag"/>
  <text class="frag-alt-kw" x="67" y="178" text-anchor="middle" dominant-baseline="central">alt</text>

  <!-- Guard: branch 1 -->
  <text class="guard-lbl" x="96" y="179" dominant-baseline="central">[payment authorized]</text>

  <!-- ─── Branch 1: payment authorized ─── -->

  <!-- ④ « authorized »  :PaymentGateway → :OrderController (dashed return) -->
  <line x1="386" y1="200" x2="294" y2="200" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="342" y="195" text-anchor="middle">« authorized »</text>

  <!-- ⑤ reserveItems(cartItems)  :OrderController → :InventorySystem -->
  <line x1="294" y1="225" x2="486" y2="225" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="392" y="220" text-anchor="middle">reserveItems(cartItems)</text>

  <!-- ⑥ « itemsReserved »  :InventorySystem → :OrderController (dashed return) -->
  <line x1="486" y1="250" x2="294" y2="250" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="392" y="245" text-anchor="middle">« itemsReserved »</text>

  <!-- ── 6. PAR COMBINED FRAGMENT (nested inside alt branch 1)  y=266 → y=373 ── -->

  <!-- Inner par rectangle -->
  <rect x="60" y="266" width="560" height="107" rx="3" class="frag-par-bg"/>

  <!-- Pentagon "par" tag -->
  <polygon points="60,266 97,266 102,272 102,284 60,284" class="frag-par-tag"/>
  <text class="frag-par-kw" x="81" y="275" text-anchor="middle" dominant-baseline="central">par</text>

  <!-- Par branch 1: email confirmation -->

  <!-- ⑦ sendConfirmationEmail()  :OrderController → :EmailService -->
  <line x1="294" y1="295" x2="586" y2="295" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="442" y="290" text-anchor="middle">sendConfirmationEmail()</text>

  <!-- ⑧ « emailQueued »  :EmailService → :OrderController (dashed return) -->
  <line x1="586" y1="318" x2="294" y2="318" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="442" y="313" text-anchor="middle">« emailQueued »</text>

  <!-- Par branch divider (dashed, per UML spec) -->
  <line x1="60" y1="336" x2="620" y2="336" class="frag-par-div"/>

  <!-- Par branch 2: stock level update -->

  <!-- ⑨ updateStockLevels()  :OrderController → :InventorySystem -->
  <line x1="294" y1="355" x2="486" y2="355" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="392" y="350" text-anchor="middle">updateStockLevels()</text>

  <!-- PAR fragment ends at y=373 -->

  <!-- ⑩ « orderPlaced »  :OrderController → :Customer (dashed return, after par) -->
  <line x1="286" y1="395" x2="90"  y2="395" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="190" y="390" text-anchor="middle">« orderPlaced »</text>

  <!-- ─── Alt else: [payment failed] ─── -->

  <!-- Alt branch divider 1 (solid line) -->
  <line x1="45" y1="415" x2="635" y2="415" class="frag-alt-div"/>
  <text class="guard-lbl" x="50" y="429" dominant-baseline="central">[payment failed]</text>

  <!-- ⑪ « authFailed »  :PaymentGateway → :OrderController (dashed return) -->
  <line x1="390" y1="448" x2="294" y2="448" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="344" y="443" text-anchor="middle">« authFailed »</text>

  <!-- ⑫ error(PAYMENT_FAILED)  :OrderController → :Customer -->
  <line x1="286" y1="470" x2="90"  y2="470" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="190" y="465" text-anchor="middle">error(PAYMENT_FAILED)</text>

  <!-- ─── Alt else: [item unavailable] ─── -->

  <!-- Alt branch divider 2 (solid line) -->
  <line x1="45" y1="490" x2="635" y2="490" class="frag-alt-div"/>
  <text class="guard-lbl" x="50" y="504" dominant-baseline="central">[item unavailable]</text>

  <!-- ⑬ « unavailable »  :InventorySystem → :OrderController (dashed return) -->
  <line x1="486" y1="523" x2="294" y2="523" class="msg-ret" marker-end="url(#arr-ret)"/>
  <text class="rlbl" x="392" y="518" text-anchor="middle">« unavailable »</text>

  <!-- ⑭ error(ITEM_UNAVAILABLE)  :OrderController → :Customer -->
  <line x1="286" y1="545" x2="90"  y2="545" class="msg-call" marker-end="url(#arr-call)"/>
  <text class="mlbl" x="190" y="540" text-anchor="middle">error(ITEM_UNAVAILABLE)</text>

  <!-- ALT fragment ends at y=563 -->

  <!-- ── 7. LIFELINE END CAPS (short horizontal tick at y=590) ── -->
  <line x1="83"  y1="590" x2="97"  y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>
  <line x1="183" y1="590" x2="197" y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>
  <line x1="283" y1="590" x2="297" y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>
  <line x1="383" y1="590" x2="397" y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>
  <line x1="483" y1="590" x2="497" y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>
  <line x1="583" y1="590" x2="597" y2="590" stroke="var(--text-tertiary)" stroke-width="1.5"/>

  <!-- ── 8. LEGEND ── -->
  <text class="ts" x="45" y="612" opacity=".45">Legend —</text>

  <line x1="110" y1="609" x2="148" y2="609"
        stroke="var(--text-primary)" stroke-width="1.5" marker-end="url(#arr-call)"/>
  <text class="ts" x="154" y="613" opacity=".75">Synchronous call</text>

  <line x1="288" y1="609" x2="326" y2="609"
        stroke="var(--text-secondary)" stroke-width="1.5"
        stroke-dasharray="5 3" marker-end="url(#arr-ret)"/>
  <text class="ts" x="332" y="613" opacity=".75">Return message</text>

  <rect x="458" y="603" width="22" height="13" rx="2"
        fill="#FAEEDA" fill-opacity="0.5" stroke="#854F0B" stroke-width="0.75"/>
  <text class="ts" x="484" y="613" opacity=".75">alt fragment</text>

  <rect x="558" y="603" width="22" height="13" rx="2"
        fill="#E1F5EE" fill-opacity="0.6" stroke="#0F6E56" stroke-width="0.75"/>
  <text class="ts" x="584" y="613" opacity=".75">par fragment</text>

  <!-- Message group annotation -->
  <text class="ts" x="45" y="632" opacity=".35">
    ①–③ pre-condition  ·  ④–⑩ happy path  ·  ⑪–⑫ payment failure  ·  ⑬–⑭ item unavailable
  </text>

</svg>
```

## Custom CSS

Add these classes to the hosting page `<style>` block (in addition to the standard skill CSS):

```css
/* ── Actor lifeline header boxes ── */
.actor       { fill: var(--bg-secondary); stroke: var(--text-secondary); stroke-width: 0.5; }
.actor-name  { font-family: system-ui, sans-serif; font-size: 11.5px; font-weight: 600;
               fill: var(--text-primary); }
.actor-colon { font-family: system-ui, sans-serif; font-size: 10px; fill: var(--text-tertiary); }

/* ── Lifeline dashed stems ── */
.lifeline { stroke: var(--text-tertiary); stroke-width: 1; stroke-dasharray: 6 4; fill: none; }

/* ── Activation bars ── */
.activation { fill: var(--bg-secondary); stroke: var(--text-secondary); stroke-width: 0.75; }

/* ── Message arrows ── */
.msg-call { stroke: var(--text-primary);   stroke-width: 1.5; fill: none; }
.msg-ret  { stroke: var(--text-secondary); stroke-width: 1.5; fill: none; stroke-dasharray: 6 3; }

/* ── Message labels ── */
.mlbl { font-family: system-ui, sans-serif; font-size: 11px; fill: var(--text-primary); }
.rlbl { font-family: system-ui, sans-serif; font-size: 11px; fill: var(--text-secondary);
        font-style: italic; }

/* ── Combined fragment: alt (amber) ── */
.frag-alt-bg  { fill: #FAEEDA; fill-opacity: 0.18; stroke: #854F0B; stroke-width: 1; }
.frag-alt-tag { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.75; }
.frag-alt-kw  { font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700;
                fill: #633806; }
.frag-alt-div { stroke: #854F0B; stroke-width: 0.75; fill: none; }
.guard-lbl    { font-family: system-ui, sans-serif; font-size: 10.5px; font-style: italic;
                fill: #854F0B; }

/* ── Combined fragment: par (teal) ── */
.frag-par-bg  { fill: #E1F5EE; fill-opacity: 0.35; stroke: #0F6E56; stroke-width: 1; }
.frag-par-tag { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.75; }
.frag-par-kw  { font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700;
                fill: #085041; }
.frag-par-div { stroke: #0F6E56; stroke-width: 0.75; stroke-dasharray: 5 3; fill: none; }

/* ── Dark mode overrides ── */
@media (prefers-color-scheme: dark) {
  .actor       { fill: #2c2c2a; stroke: #b4b2a9; }
  .actor-name  { fill: #e8e6de; }
  .actor-colon { fill: #888780; }
  .frag-alt-bg  { fill: #633806; fill-opacity: 0.25; stroke: #EF9F27; }
  .frag-alt-tag { fill: #633806; stroke: #EF9F27; }
  .frag-alt-kw  { fill: #FAC775; }
  .frag-alt-div { stroke: #EF9F27; }
  .guard-lbl    { fill: #EF9F27; }
  .frag-par-bg  { fill: #085041; fill-opacity: 0.35; stroke: #5DCAA5; }
  .frag-par-tag { fill: #085041; stroke: #5DCAA5; }
  .frag-par-kw  { fill: #9FE1CB; }
  .frag-par-div { stroke: #5DCAA5; }
}
```

## Color Assignments

| Element | Color | Reason |
|---------|-------|--------|
| Actor header boxes | Neutral (`var(--bg-secondary)`) | Structural / non-semantic — all lifelines share one style |
| Activation bars | Neutral (`var(--bg-secondary)`) | Show execution periods without adding semantic color |
| Synchronous call arrows | `var(--text-primary)` + filled triangle | High contrast for calls — the primary interaction direction |
| Return / dashed arrows | `var(--text-secondary)` + open chevron | Lower contrast for returns — secondary flow direction |
| `alt` fragment | Amber (`#FAEEDA` / `#854F0B`) | Warning / conditional — matches `c-amber` semantic meaning |
| Guard condition text | Amber italic | Belongs visually to the alt fragment |
| `par` fragment | Teal (`#E1F5EE` / `#0F6E56`) | Concurrent success path — matches `c-teal` semantic meaning |
| Alt branch dividers | Amber solid line | Continuity with the alt frame color |
| Par branch divider | Teal dashed line | UML spec: par branches separated by dashed lines |

## Layout Notes

- **ViewBox**: 680×648 (standard width; height = lifeline bottom y=590 + legend + annotation + 16px buffer)
- **Lifeline spacing formula**: `(safe_area_width) / (n_lifelines − 1) = 600 / 5 = 120px` — but use `spacing = 100px` starting at `x=90` so that first box left = 40 and last box right = 640 exactly
- **Actor box split-label trick**: Two separate `<text>` elements per box — one for `":"` (10px, tertiary color) and one for the class name (11.5px bold, primary color) — avoids the 14px font needing ~150px+ per box for long names like "OrderController"
- **Pentagon tag formula**: For a fragment starting at `(fx, fy)`, the tag polygon points are `(fx,fy) (fx+w,fy) (fx+w+6,fy+6) (fx+w+6,fy+18) (fx,fy+18)` where `w` = approximate text width of the keyword + 8px padding each side
- **Nested fragment inset**: The `par` rect uses `x = alt_x + 15` and `y = alt_y_current + 2` so both borders remain simultaneously visible — inset enough to separate visually, not so much that it wastes vertical space
- **Activation bar placement**: `x = lifeline_cx − 4`, `width = 8` — centered on the lifeline and narrow enough not to obscure the dashed stem behind it
- **Message label y-offset**: All labels are placed at `y = arrow_y − 5` to sit just above the arrow line; this applies to both left-going and right-going arrows since `text-anchor="middle"` handles horizontal centering automatically
- **Return arrows entering activation bars**: End `x1/x2` at lifeline center (e.g. x=294 for OrderController) rather than the bar edge (x=286) — the small overlap is intentional and clarifies the target object
- **Alt guard label placement**: Branch 1 guard goes at `y = frame_top + 13` to the right of the pentagon tag; subsequent branch guards go at `divider_y + 14` so they sit just inside the new branch
- **Lifeline end cap pattern**: `<line x1="cx−7" y1="590" x2="cx+7" y2="590" stroke-width="1.5"/>` — a simple symmetric tick, no special marker needed
