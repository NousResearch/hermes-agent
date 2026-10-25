export const SCREEN = (() => {
  const W = 224;
  const H = 288;
  const FONT = {
    A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    C: ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    I: ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    0: ["01110", "10011", "10101", "10101", "10101", "11001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
  };
  const SPRITES = {
    jellyA: ["0001111000", "0011111100", "0110110110", "1111111111", "1111111111", "0010110100", "0100000010", "0010000100"],
    jellyB: ["0001111000", "0011111100", "0110110110", "1111111111", "1111111111", "0010110100", "0001001000", "0010000100"],
    eyeA: ["0011110000", "0111111000", "1101101100", "1111111100", "0110011000", "1100001100"].map((r) => r.slice(0, 8)),
    eyeB: ["0011110000", "0111111000", "1101101100", "1111111100", "0110011000", "0011110000"].map((r) => r.slice(0, 8)),
    ship: ["00000100000", "00001110000", "00001110000", "01111111110", "11111111111", "11011111011", "10000100001"],
    boom: ["1000100010", "0100100100", "0010001000", "1100000011", "0010001000", "0100100100", "1000100010"],
  };
  const n = (v) => Math.round(v * 100) / 100;
  function bitmap(rows, x, y, s = 1) {
    let d = "";
    rows.forEach((row, r) => {
      let c = 0;
      while (c < row.length) {
        if (row[c] === "1") {
          let e = c;
          while (e < row.length && row[e] === "1") e++;
          d += `M${n(x + c * s)} ${n(y + r * s)}h${n((e - c) * s)}v${n(s)}h${n(-(e - c) * s)}z`;
          c = e;
        } else c++;
      }
    });
    return d;
  }
  function text(str, x, y, s = 1, adv = 8) {
    return [...String(str)].map((ch, i) => bitmap(FONT[ch] ?? FONT[" "], x + i * adv * s, y, s)).join("");
  }
  const centred = (str, s = 1, adv = 8) => n((W - (String(str).length * adv - (adv - 5)) * s) / 2);

  const FORMATION = [];
  for (let row = 0; row < 3; row++) for (let col = 0; col < 6; col++) FORMATION.push({ row, col, x: 34 + col * 26, y: 92 + row * 18, kind: row === 0 ? "eye" : "jelly" });
  const SHIP_Y = 250;
  const px = (d, extra = "") => `<path class="ac-px"${extra} d="${d}"/>`;

  function svg() {
    const out = [];
    const hatch = [];
    for (let x = 0; x <= W; x += 16) hatch.push(`M${Math.min(x, W - 1)} 0h1v${H}h-1z`);
    for (let y = 0; y <= H; y += 16) hatch.push(`M0 ${Math.min(y, H - 1)}h${W}v1h${-W}z`);
    out.push(`<g data-mode="test" style="display:none">`);
    out.push(`<path class="ac-px ac-dim" d="${hatch.join("")}"/>`);
    out.push(px(text("SELF TEST", centred("SELF TEST"), 56), ` data-step="0"`));
    out.push(px(text("RAM OK", 56, 104), ` data-step="1"`));
    out.push(px(text("ROM OK", 56, 120), ` data-step="2"`));
    out.push(px(text("Z80 3.072 MHZ", 56, 152), ` data-step="3"`));
    out.push(`</g>`);
    out.push(`<g data-mode="attract" style="display:none">`);
    out.push(px(text("1UP", 24, 8) + text("HI-SCORE", 80, 8) + text("2UP", 176, 8)));
    out.push(px(text("00000", 16, 20), ` data-part="score"`));
    out.push(px(text("12500", 92, 20)));
    out.push(px(text("ANATOMY", centred("ANATOMY", 2), 44, 2)));
    out.push(`<g data-part="formation">`);
    FORMATION.forEach((a, i) => {
      const [fa, fb] = a.kind === "eye" ? [SPRITES.eyeA, SPRITES.eyeB] : [SPRITES.jellyA, SPRITES.jellyB];
      const ox = a.kind === "eye" ? 1 : 0;
      out.push(`<g data-alien="${i}">${px(bitmap(fa, a.x + ox, a.y), ' data-frame="a"')}${px(bitmap(fb, a.x + ox, a.y), ' data-frame="b" style="display:none"')}</g>`);
    });
    out.push(`</g>`);
    out.push(px(bitmap(SPRITES.boom, 0, 0), ` data-part="boom" style="display:none"`));
    out.push(px("M0 0h1v5h-1z", ` data-part="shot" style="display:none"`));
    out.push(px(bitmap(SPRITES.ship, -5.5, 0), ` data-part="ship" transform="translate(112 ${SHIP_Y})"`));
    out.push(px(`M8 ${SHIP_Y + 12}h${W - 16}v1h${-(W - 16)}z`));
    out.push(px(text("INSERT COIN", centred("INSERT COIN"), 200), ` data-part="insert"`));
    out.push(px(text("PUSH START", centred("PUSH START"), 200), ` data-part="start" style="display:none"`));
    out.push(px(text("CREDIT 0", 120, 274), ` data-part="credit"`));
    out.push(`</g>`);
    const k = 58 / W;
    const scan = [];
    for (let i = 0; i <= 112; i++) scan.push(`M${n(-29 + i * 0.518)} -37.3V37.3`);
    return [
      `<g data-part="raster"><g transform="translate(-29 -37.3) scale(${n(k * 10000) / 10000})">${out.join("")}</g>`,
      `<path class="ac-scan" d="${scan.join("")}"/></g>`,
      `<ellipse class="ac-dot" data-part="dot" cx="0" cy="0" rx="1.2" ry="1.2" style="opacity:0"/>`,
    ].join("");
  }
  return { W, H, FONT, SPRITES, FORMATION, SHIP_Y, bitmap, text, centred, svg };
})();
