
export function makeParts(k, G, P, D) {
  const V = G.viewOf(P);
  const at = (p) => k.iso(p, P);
  const S = (paths, style) => k.solidSvg(paths, style);
  const Ln = (d, style) => k.lineSvg(d, style);
  const rot = (v, axis, angle) => {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const d = G.dot3(axis, v);
    const x = G.cross3(axis, v);
    return [v[0] * c + x[0] * s + axis[0] * d * (1 - c), v[1] * c + x[1] * s + axis[1] * d * (1 - c), v[2] * c + x[2] * s + axis[2] * d * (1 - c)];
  };
  const J = D.stick;
  const UP = J.nc;
  const ALONG = J.dc;
  const X = [1, 0, 0];
  const axisOf = (tx, tv) => G.unit3(rot(rot(UP, ALONG, tx), X, -tv));
  const pivot = G.add3(J.o, G.mul3(UP, J.pivot));
  const depth = (p) => G.dot3(p, V);

  function stickUpper(tx = 0, tv = 0) {
    const a = axisOf(tx, tv);
    const F = G.frameAlong(pivot, a);
    const t = -J.pivot / G.dot3(a, UP);
    const cross = G.add3(pivot, G.mul3(a, t));
    const W = G.frameOf(cross, UP, X, ALONG);
    const washer = S(G.disc(0, 0.4, 4.4, W, P, { steps: 40 }), { tone: "mid" }) + Ln(G.circleOf(W, 0.4, 1.3, P, 20), { tone: "faint" });
    const shaft = S(G.disc(t + 0.4, t + 8.95, 0.85, F, P, { steps: 16 }), { tone: "mid" });
    const ballAt = G.add3(pivot, G.mul3(a, t + 12.4));
    const ball = S(G.sphereOf(ballAt, 3.5, P, { steps: 40 }), { tone: "hi", lit: true });
    return washer + shaft + ball;
  }

  function stickLower(tx = 0, tv = 0) {
    const a = axisOf(tx, tv);
    const F = G.frameAlong(pivot, a);
    const parts = [];
    parts.push({ d: depth(G.add3(pivot, G.mul3(a, -5))), svg: S(G.disc(-9.4, -0.6, 0.85, F, P, { steps: 16 }), { tone: "mid" }) });
    const act = S(G.disc(-8.6, -4.6, 2.4, F, P, { steps: 32, bevel: 0.3 }), { tone: "hi" }) + Ln(G.arcOf(F, -6.6, 2.4, P), { tone: "faint" });
    parts.push({ d: depth(G.add3(pivot, G.mul3(a, -6.6))) + 0.01, svg: act });
    parts.push({ d: depth(G.add3(pivot, G.mul3(a, -9.6))), svg: S(G.disc(-10, -9.4, 1.5, F, P, { steps: 20 }), { tone: "lo" }) });
    const closed = switchesClosed(tx, tv);
    for (const sw of J.switches) {
      const on = closed.includes(sw.id);
      const body = G.frameOf(sw.c, UP, sw.out, G.cross3(UP, sw.out));
      const poly = [[-0.65, -2], [0.65, -2], [0.65, 2], [-0.65, 2]];
      const box = S(G.prismOf(body, poly, -1, 1, P), { tone: "mid" });
      const lever0 = G.add3(sw.c, G.add3(G.mul3(sw.out, -0.65), G.mul3(G.cross3(UP, sw.out), 1.6)));
      const tip = G.add3(sw.c, G.add3(G.mul3(sw.out, on ? -0.85 : -1.45), G.mul3(G.cross3(UP, sw.out), -1.2)));
      const lever = Ln(k.pathOf([at(lever0), at(tip)]), { tone: on ? "lit" : "hi" });
      const tabs = [-1.1, 0, 1.1].map((o) => G.add3(sw.c, G.add3(G.mul3(UP, -1), G.mul3(G.cross3(UP, sw.out), o))));
      const tabSvg = tabs.map((p) => S(G.prismOf(G.frameOf(p, UP, sw.out, G.cross3(UP, sw.out)), [[-0.3, -0.35], [0.3, -0.35], [0.3, 0.35], [-0.3, 0.35]], -1, 0, P), { tone: "lo" })).join("");
      parts.push({ d: depth(sw.c), svg: `<g class="ac-switch"${on ? " data-on" : ""}>${tabSvg}${box}${lever}</g>` });
    }
    parts.sort((p, q) => p.d - q.d);
    return parts.map((p) => p.svg).join("");
  }

  function switchesClosed(tx, tv) {
    const out = [];
    const limit = 0.12;
    if (tx > limit) out.push("left");
    if (tx < -limit) out.push("right");
    if (tv > limit) out.push("up");
    if (tv < -limit) out.push("down");
    return out;
  }

  const COIN_R = 2.426;
  const COIN_T = 0.35;
  function coin(centre, axis, spin = 0, { lit = true } = {}) {
    const F0 = G.frameAlong(centre, axis);
    const F = { o: F0.o, a: F0.a, u: rot(F0.u, F0.a, spin), v: rot(F0.v, F0.a, spin) };
    const face = G.dot3(F.a, V) >= 0 ? COIN_T / 2 : -COIN_T / 2;
    const body = S(G.disc(-COIN_T / 2, COIN_T / 2, COIN_R, F, P, { steps: 36 }), { tone: "hi", lit });
    const facing = Math.abs(G.dot3(F.a, V));
    let marks = "";
    if (facing > 0.2) {
      marks += Ln(G.circleOf(F, face, COIN_R - 0.35, P, 28), { tone: "lo" });
      const p0 = G.pointOf(F, face, 0.3, 0);
      const p1 = G.pointOf(F, face, 1.5, 0);
      marks += Ln(k.pathOf([at(p0), at(p1)]), { tone: "lo", free: true });
    }
    return body + marks;
  }

  return { stickUpper, stickLower, switchesClosed, coin, axisOf, COIN_R, COIN_T };
}
