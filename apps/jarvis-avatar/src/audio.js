/* Áudio: microfone, <audio>/<video> externo e a voz do navegador → volume e 8 bandas de frequência. */
import { CFG } from './config.js';
import { ease, toast } from './util.js';

const BAND_EDGES = [70, 160, 320, 640, 1100, 1900, 3300, 6200, 11000];
const FORMANT = [0.85, 1.0, 0.95, 0.8, 0.62, 0.48, 0.34, 0.22];

// clock.time: segundos desde o início (vem do loop) · onOnset(): ataque forte da voz · micButton: alterna aria-pressed
export function createAudio({ clock, onOnset, micButton }) {
  const A = { ctx: null, an: null, td: null, fd: null, mic: null, stream: null, media: new Set(),
              level: 0, prev: 0, bands: new Float32Array(8), manual: 0, onsetT: 0 };
  const rawBands = new Float32Array(8), synthBands = new Float32Array(8);

  function audioCtx() {
    if (!A.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error('Web Audio indisponível neste navegador');
      A.ctx = new AC();
      A.an = A.ctx.createAnalyser();
      A.an.fftSize = 1024;
      A.an.smoothingTimeConstant = 0.5;
      A.td = new Uint8Array(A.an.fftSize);
      A.fd = new Uint8Array(A.an.frequencyBinCount);
    }
    if (A.ctx.state === 'suspended') A.ctx.resume();
    return A.ctx;
  }
  function analyse() {
    A.an.getByteTimeDomainData(A.td);
    A.an.getByteFrequencyData(A.fd);
    let sum = 0;
    for (let i = 0; i < A.td.length; i++) { const v = (A.td[i] - 128) / 128; sum += v * v; }
    const binHz = A.ctx.sampleRate / A.an.fftSize;
    for (let b = 0; b < 8; b++) {
      const i0 = Math.max(1, Math.floor(BAND_EDGES[b] / binHz));
      const i1 = Math.min(A.fd.length, Math.max(i0 + 1, Math.floor(BAND_EDGES[b + 1] / binHz)));
      let acc = 0;
      for (let i = i0; i < i1; i++) acc += A.fd[i];
      rawBands[b] = Math.min(1, Math.pow(acc / ((i1 - i0) * 255), 1.3) * 1.8);
    }
    return Math.min(1, Math.sqrt(sum / A.td.length) * 4.5);
  }
  function mediaPlaying() {
    for (const el of A.media) if (!el.paused && !el.ended) return true;
    return false;
  }

  // A voz do navegador não expõe o áudio: enquanto ela fala, o volume é um envelope SIMULADO de sílabas.
  // Com o teu TTS real, use JARVIS.attachAudio(el) e a boca segue o áudio de verdade.
  const speech = { active: false, kick: 0, timer: 0, current: null };
  function synthVoice(t) {
    const syl = Math.abs(Math.sin(t * 7.2 + 1.6 * Math.sin(t * 2.3)));
    const phrase = 0.55 + 0.45 * Math.sin(t * 0.9) * Math.sin(t * 0.41 + 1.0);
    const lvl = Math.min(1, (0.16 + 0.84 * syl * syl * (0.55 + 0.45 * phrase)) * 0.78 + speech.kick * 0.35);
    for (let b = 0; b < 8; b++) synthBands[b] = lvl * (0.45 + 0.55 * Math.abs(Math.sin(t * (2.6 + b * 0.9) + b * 1.9))) * FORMANT[b];
    return lvl;
  }
  function update(dt) {
    rawBands.fill(0);
    synthBands.fill(0);
    const time = clock.time;
    const micLvl = A.ctx && (A.mic || mediaPlaying()) ? analyse() : 0;
    speech.kick *= Math.exp(-dt * 8);
    const synLvl = speech.active ? synthVoice(time) : 0;
    const tgt = Math.max(micLvl, synLvl, A.manual);
    A.level += (tgt - A.level) * ease(dt, tgt > A.level ? 28 : 7);
    for (let b = 0; b < 8; b++) {
      const tb = Math.max(rawBands[b], synthBands[b], A.manual * 0.7);
      A.bands[b] += (tb - A.bands[b]) * ease(dt, tb > A.bands[b] ? 30 : 8);
    }
    if (A.level - A.prev > 0.12 && A.level > 0.25 && time - A.onsetT > 0.22) {
      onOnset();
      A.onsetT = time;
    }
    A.prev = A.level;
  }

  let voices = [];
  const loadVoices = () => { voices = window.speechSynthesis ? speechSynthesis.getVoices() : []; };
  if (window.speechSynthesis) {
    loadVoices();
    if (speechSynthesis.addEventListener) speechSynthesis.addEventListener('voiceschanged', loadVoices);
  }
  function pickVoice() {
    const pt = voices.filter((v) => /^pt[-_]?BR/i.test(v.lang));
    return pt.find((v) => /felipe|daniel|google/i.test(v.name)) || pt[0] || voices.find((v) => /^pt/i.test(v.lang)) || null;
  }
  function animateSpeechOnly(text) {
    speech.active = true;
    clearTimeout(speech.timer);
    speech.timer = setTimeout(() => { speech.active = false; }, Math.max(1500, text.length * 70));
  }
  function speak(raw) {
    const text = String(raw || '').trim() || CFG.defaultPhrase;
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
      animateSpeechOnly(text);
      toast('Este navegador não tem voz sintetizada; mostrando só a animação de fala.');
      return;
    }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    speech.current = u;
    const isCurrent = () => speech.current === u;
    const v = pickVoice();
    u.lang = v ? v.lang : CFG.voiceLang;
    if (v) u.voice = v;
    u.rate = 1.03;
    u.pitch = 0.85;
    let started = false;
    u.onstart = () => { if (!isCurrent()) return; started = true; speech.active = true; clearTimeout(speech.timer); };
    u.onend = () => { if (isCurrent()) speech.active = false; };
    u.onerror = (ev) => {
      if (!isCurrent() || ev.error === 'canceled' || ev.error === 'interrupted') return;
      speech.active = false;
      animateSpeechOnly(text);
      toast(`A voz do navegador falhou (${ev.error}); mostrando só a animação de fala.`);
    };
    u.onboundary = () => { speech.kick = 1; };
    speechSynthesis.speak(u);
    setTimeout(() => {
      if (isCurrent() && !started && !speechSynthesis.speaking) {
        animateSpeechOnly(text);
        toast('O navegador não liberou a voz aqui; mostrando só a animação de fala.');
      }
    }, 1200);
  }

  function stopMic() {
    if (A.stream) A.stream.getTracks().forEach((t) => t.stop());
    if (A.mic) A.mic.disconnect();
    A.stream = null;
    A.mic = null;
    micButton.setAttribute('aria-pressed', 'false');
  }
  async function toggleMic() {
    if (A.mic) { stopMic(); toast('Microfone desligado.'); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast('Microfone indisponível aqui. Abra a página em https ou em http://localhost.', 6500);
      return;
    }
    try {
      audioCtx();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      A.stream = stream;
      A.mic = A.ctx.createMediaStreamSource(stream);
      A.mic.connect(A.an);
      micButton.setAttribute('aria-pressed', 'true');
      toast('Microfone ligado. Fale e veja as partículas responderem.');
    } catch (err) {
      stopMic();
      const why = err && err.name === 'NotAllowedError' ? 'a permissão foi negada ou bloqueada' : (err && err.message) || String(err);
      toast(`Não deu pra abrir o microfone (${why}). Se estiver no preview, abra a página direto no navegador.`, 7000);
    }
  }

  // Áudio de outra origem só é analisável com CORS (crossorigin="anonymous" + Access-Control-Allow-Origin);
  // sem isso o navegador entrega silêncio ao analisador e a boca fica parada.
  function attachAudio(el) {
    if (!(el instanceof HTMLMediaElement)) throw new TypeError('JARVIS.attachAudio espera um elemento <audio> ou <video>');
    audioCtx();
    if (!A.media.has(el)) {
      const src = A.ctx.createMediaElementSource(el);
      src.connect(A.an);
      src.connect(A.ctx.destination);
      A.media.add(el);
    }
  }

  return { A, speech, update, speak, toggleMic, attachAudio };
}
