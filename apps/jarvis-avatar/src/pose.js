/* Pose da cabeça: o que faz ele parecer vivo (balanço, olhadas, seguir o mouse, reagir aos estados). */
import { IMG, STILL, REDUCED } from './config.js';
import { clamp, ease, rnd } from './util.js';

// Movimento fluido: cada eixo passa por dois estágios. O alvo é suavizado (filtro de 1ª ordem) e só então
// puxa uma mola criticamente amortecida. Assim todo movimento começa e termina em curva: sem tranco na
// saída (a aceleração cresce aos poucos) e sem passar do ponto na chegada.
const axis = (filt, w) => ({ x: 0, v: 0, t: 0, filt, w });
const POSE_STEP = 1 / 120;                                    // passo fixo: o mesmo movimento a 30, 60 ou 120 fps
const POSE_LIM = { yaw: 0.46, pitch: 0.26, roll: 0.06 };      // giro 3D até ~26°; acima disso a lateral da cabeça,
                                                              // que a foto de frente quase não mostra, fica artificial
function drive(s, target, dt) {
  for (let left = dt; left > 1e-6; left -= POSE_STEP) {
    const h = Math.min(POSE_STEP, left);
    s.t += (target - s.t) * (1 - Math.exp(-s.filt * h));
    s.v += (s.w * s.w * (s.t - s.x) - 2 * s.w * s.v) * h;
    s.x += s.v * h;
  }
}
function springLoose(s, target, w, zeta, dt) {
  s.v += (w * w * (target - s.x) - 2 * zeta * w * s.v) * dt;
  s.x += s.v * dt;
}

// clock.time · mouse: estado do ponteiro (px da imagem) · ST: estados suavizados · audio.A.level: volume
export function createPose({ clock, mouse, ST, audio }) {
  const pose = { yaw: axis(5, 5.5), pitch: axis(5, 5.5), roll: axis(3, 3.6),
                 leanX: axis(3, 3.2), leanY: axis(3, 3.2), zoom: axis(2.5, 2.8),
                 breath: 0, talk: 0, glanceAt: 4, glanceUntil: 0, gYaw: 0, gPitch: 0 };
  // a mesma pose atrás de uma mola mole: a aura vem atrás da cabeça e balança
  const lag = { yaw: { x: 0, v: 0 }, pitch: { x: 0, v: 0 }, roll: { x: 0, v: 0 },
                leanX: { x: 0, v: 0 }, leanY: { x: 0, v: 0 }, zoom: { x: 0, v: 0 } };

  // ruído suave: soma de senos com frequências que não se repetem juntas
  const wobble = (a, b, c, ph) => {
    const time = clock.time;
    return Math.sin(time * a + ph) * 0.6 + Math.sin(time * b + ph * 2.3) * 0.3 + Math.sin(time * c + ph * 3.7) * 0.1;
  };

  function update(dt) {
    if (STILL || REDUCED) return;                       // repouso exato (comparação com a foto) e acessibilidade
    const time = clock.time;
    let ty = 0.04 * wobble(0.23, 0.41, 0.97, 0.0);    // em repouso: balanço lento, nunca parado
    let tp = 0.022 * wobble(0.19, 0.33, 0.81, 1.3);
    let tr = 0.010 * wobble(0.15, 0.27, 0.66, 2.1);
    let ly = 0, lz = 0;                                 // avanço: só quando há mouse
    // olhadas: de tempos em tempos olha pra um lado e volta, só quando ninguém mexe nele
    if (mouse.active) pose.glanceAt = Math.max(pose.glanceAt, time + 3);
    else if (time > pose.glanceAt) {
      pose.gYaw = (rnd() < 0.5 ? -1 : 1) * (0.15 + rnd() * 0.15);
      pose.gPitch = (rnd() - 0.5) * 0.06;
      pose.glanceUntil = time + 1.2 + rnd() * 1.6;
      pose.glanceAt = pose.glanceUntil + 4 + rnd() * 6;
    }
    if (time < pose.glanceUntil) { ty += pose.gYaw; tp += pose.gPitch; }
    if (mouse.active) {                                 // segue o ponteiro com a cabeça (posição suavizada: sem tranco)
      const mx = mouse.seen ? mouse.fx : mouse.ix, my = mouse.seen ? mouse.fy : mouse.iy;
      ty = ty * 0.4 + clamp((mx - IMG.axisX) / 620, -1, 1) * 0.46;   // vira a cabeça na direção do mouse (giro 3D)
      const dyN = clamp((my - IMG.headCY) / 520, -1, 1);
      tp = tp * 0.4 + dyN * (dyN > 0 ? 0.16 : 0.24);
      tr *= 0.4;
      // quanto mais perto o cursor, mais ele chega perto da tela (sem deslocar pro lado: volta junto com o giro)
      const dist = Math.hypot(mx - IMG.axisX, (my - IMG.headCY) * 1.2);
      const near = 1 - clamp((dist - 140) / 560, 0, 1);
      ly = dyN * (dyN > 0 ? 3 : 5);
      lz = 0.07 * near * near * (3 - 2 * near);
    }
    tr += ST.listen * 0.04; tp += ST.listen * 0.012;                                   // ouvindo: inclina curioso
    tp -= ST.think * 0.028; tr -= ST.think * 0.012; ty += ST.think * 0.06 * Math.sin(time * 0.55);   // pensando: olha pra cima
    // falando: acena num ritmo calmo; a força do aceno segue o volume já suavizado (o volume cru treme)
    pose.talk += (audio.A.level - pose.talk) * ease(dt, 4);
    tp += pose.talk * 0.04 * Math.sin(time * 4.2);
    ty += pose.talk * 0.03 * Math.sin(time * 1.7 + 0.8);
    drive(pose.yaw, clamp(ty, -POSE_LIM.yaw, POSE_LIM.yaw), dt);
    drive(pose.pitch, clamp(tp, -POSE_LIM.pitch, POSE_LIM.pitch), dt);
    drive(pose.roll, clamp(tr, -POSE_LIM.roll, POSE_LIM.roll), dt);
    drive(pose.leanX, 0, dt);                            // o corpo tem peso: avança mais devagar que o olhar
    drive(pose.leanY, ly, dt);
    drive(pose.zoom, lz, dt);
    pose.breath = Math.sin(time * 2 * Math.PI / 4.6) * (1 + 0.5 * ST.speak);           // respiração: ~13 por minuto
    for (const k in lag) springLoose(lag[k], pose[k].x, 2.6, 0.38, dt);
  }

  function uniforms(gl, p) {
    gl.uniform4f(p.u.uPose, 0, pose.pitch.x, pose.roll.x, pose.breath);   // o giro (yaw) vai pelo uYaw, em 3D
    gl.uniform2f(p.u.uYaw, pose.yaw.x, lag.yaw.x);
    gl.uniform3f(p.u.uLean, pose.leanX.x, pose.leanY.x, pose.zoom.x);
    if (p.u.uPoseLag) gl.uniform4f(p.u.uPoseLag, 0, lag.pitch.x, lag.roll.x, pose.breath);
    if (p.u.uLeanLag) gl.uniform3f(p.u.uLeanLag, lag.leanX.x, lag.leanY.x, lag.zoom.x);
  }

  return { pose, update, uniforms };
}
