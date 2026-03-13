// =========================================================================
// Hermes Agent Landing Page — Interactions
// =========================================================================

// --- Platform install commands ---
const PLATFORMS = {
  linux: {
    command:
      "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash",
    prompt: "$",
    note: "Works on Linux, macOS & WSL2 · No prerequisites · Installs everything automatically",
    stepNote:
      "Installs uv, Python 3.11, clones the repo, sets up everything. No sudo needed.",
  },
};

function detectPlatform() {
  return "linux";
}

function switchPlatform(platform) {
  const cfg = PLATFORMS[platform];
  if (!cfg) return;

  // Update hero install widget
  const commandEl = document.getElementById("install-command");
  const promptEl = document.getElementById("install-prompt");
  const noteEl = document.getElementById("install-note");

  if (commandEl) commandEl.textContent = cfg.command;
  if (promptEl) promptEl.textContent = cfg.prompt;
  if (noteEl) noteEl.textContent = cfg.note;

  // Update active tab in hero
  document.querySelectorAll(".install-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.platform === platform);
  });

  // Sync the step section tabs too
  switchStepPlatform(platform);
}

function switchStepPlatform(platform) {
  const cfg = PLATFORMS[platform];
  if (!cfg) return;

  const commandEl = document.getElementById("step1-command");
  const copyBtn = document.getElementById("step1-copy");
  const noteEl = document.getElementById("step1-note");

  if (commandEl) commandEl.textContent = cfg.command;
  if (copyBtn) copyBtn.setAttribute("data-text", cfg.command);
  if (noteEl) noteEl.textContent = cfg.stepNote;

  // Update active tab in step section
  document.querySelectorAll(".code-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.platform === platform);
  });
}

function toggleMobileNav() {
  document.getElementById("nav-mobile").classList.toggle("open");
  document.getElementById("nav-hamburger").classList.toggle("open");
}

function toggleSpecs() {
  const wrapper = document.getElementById("specs-wrapper");
  const btn = document.getElementById("specs-toggle");
  const label = btn.querySelector(".toggle-label");
  const isOpen = wrapper.classList.contains("open");

  if (isOpen) {
    wrapper.style.maxHeight = wrapper.scrollHeight + "px";
    requestAnimationFrame(() => {
      wrapper.style.maxHeight = "0";
    });
    wrapper.classList.remove("open");
    btn.classList.remove("open");
    if (label) label.textContent = "More details";
  } else {
    wrapper.classList.add("open");
    wrapper.style.maxHeight = wrapper.scrollHeight + "px";
    btn.classList.add("open");
    if (label) label.textContent = "Less";
    wrapper.addEventListener(
      "transitionend",
      () => {
        if (wrapper.classList.contains("open")) {
          wrapper.style.maxHeight = "none";
        }
      },
      { once: true }
    );
  }
}

// --- Copy to clipboard ---
function copyInstall() {
  const text = document.getElementById("install-command").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".install-widget-body .copy-btn");
    const original = btn.querySelector(".copy-text").textContent;
    btn.querySelector(".copy-text").textContent = "Copied!";
    btn.style.color = "var(--primary-light)";
    setTimeout(() => {
      btn.querySelector(".copy-text").textContent = original;
      btn.style.color = "";
    }, 2000);
  });
}

function copyText(btn) {
  const text = btn.getAttribute("data-text");
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = "Copied!";
    btn.style.color = "var(--primary-light)";
    setTimeout(() => {
      btn.textContent = original;
      btn.style.color = "";
    }, 2000);
  });
}

// --- Scroll-triggered fade-in ---
function initScrollAnimations() {
  const elements = document.querySelectorAll(
    ".feature-card, .install-step, " +
      ".section-header, .terminal-window",
  );

  elements.forEach((el) => el.classList.add("fade-in"));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Stagger children within grids
          const parent = entry.target.parentElement;
          if (parent) {
            const siblings = parent.querySelectorAll(".fade-in");
            let idx = Array.from(siblings).indexOf(entry.target);
            if (idx < 0) idx = 0;
            setTimeout(() => {
              entry.target.classList.add("visible");
            }, idx * 60);
          } else {
            entry.target.classList.add("visible");
          }
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" },
  );

  elements.forEach((el) => observer.observe(el));
}

// --- Terminal Demo ---
const CURSOR = '<span class="terminal-cursor">█</span>';

const demoSequence = [
  { type: "prompt", text: "❯ " },
  {
    type: "type",
    text: "Research the latest approaches to GRPO training and write a summary",
    delay: 30,
  },
  { type: "pause", ms: 600 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-dim">  web_search "GRPO reinforcement learning 2026"       1.2s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  web_extract arxiv.org/abs/2402.03300                3.1s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  web_search "GRPO vs PPO ablation results"           0.9s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  web_extract huggingface.co/blog/grpo                2.8s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  write_file ~/research/grpo-summary.md               0.1s</span>',
    ],
  },
  { type: "pause", ms: 500 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-text">Done! I\'ve written a summary covering:</span>',
      "",
      '<span class="t-text">  <span class="t-green">✓</span> GRPO\'s group-relative advantage (no critic model needed)</span>',
      '<span class="t-text">  <span class="t-green">✓</span> Comparison with PPO/DPO on reasoning benchmarks</span>',
      '<span class="t-text">  <span class="t-green">✓</span> Implementation notes for Axolotl and TRL</span>',
      "",
      '<span class="t-text">Saved to</span> <span class="t-accent">~/research/grpo-summary.md</span>',
    ],
  },
  { type: "pause", ms: 2500 },

  { type: "clear" },
  { type: "prompt", text: "❯ " },
  {
    type: "type",
    text: "Review the PR at NousResearch/hermes-agent#42 and fix any issues",
    delay: 30,
  },
  { type: "pause", ms: 600 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-dim">  delegate_task "review PR #42 changes"                2.1s</span>',
    ],
  },
  { type: "pause", ms: 500 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  git diff main..pr-42                                 0.4s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  patch tools/registry.py                              0.1s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  python -m pytest tests/ -x                           3.2s</span>',
    ],
  },
  { type: "pause", ms: 400 },
  {
    type: "output",
    lines: [
      '<span class="t-dim">  git commit -m "fix: handle empty tool schemas"       0.3s</span>',
    ],
  },
  { type: "pause", ms: 500 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-text">Found 2 issues in the PR and fixed both:</span>',
      "",
      '<span class="t-text">  <span class="t-green">✓</span> Empty tool schema crash in registry.py — added guard</span>',
      '<span class="t-text">  <span class="t-green">✓</span> Missing error handling in delegate_tool.py — added try/except</span>',
      "",
      '<span class="t-text">Tests pass. Committed the fix and pushed to the PR branch.</span>',
      '<span class="t-text">I also saved a</span> <span class="t-accent">skill</span> <span class="t-text">for this PR review pattern.</span>',
    ],
  },
  { type: "pause", ms: 2500 },

  { type: "clear" },
  { type: "prompt", text: "❯ " },
  {
    type: "type",
    text: "How did we fix that Docker networking issue?",
    delay: 35,
  },
  { type: "pause", ms: 500 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-dim">  session_search "Docker networking"                   1.4s</span>',
    ],
  },
  { type: "pause", ms: 500 },
  {
    type: "output",
    lines: [
      "",
      '<span class="t-text">Found it — from a session on February 12th:</span>',
      "",
      '<span class="t-text">The containers couldn\'t reach each other because the compose</span>',
      '<span class="t-text">file was using the default bridge network. We switched to a</span>',
      '<span class="t-text">custom network with</span> <span class="t-accent">driver: overlay</span><span class="t-text">, added explicit</span>',
      '<span class="t-text">aliases, and set</span> <span class="t-accent">dns: 8.8.8.8</span> <span class="t-text">as a fallback.</span>',
      "",
      '<span class="t-text">The fix was committed in</span> <span class="t-accent">docker-compose.prod.yml</span><span class="t-text">.</span>',
    ],
  },
  { type: "pause", ms: 3000 },
];

class TerminalDemo {
  constructor(container) {
    this.container = container;
    this.running = false;
    this.content = "";
  }

  async start() {
    if (this.running) return;
    this.running = true;

    while (this.running) {
      for (const step of demoSequence) {
        if (!this.running) return;
        await this.execute(step);
      }
      this.clear();
      await this.sleep(1000);
    }
  }

  stop() {
    this.running = false;
  }

  async execute(step) {
    switch (step.type) {
      case "prompt":
        this.append(`<span class="t-prompt">${step.text}</span>`);
        break;
      case "type":
        for (const char of step.text) {
          if (!this.running) return;
          this.append(`<span class="t-cmd">${char}</span>`);
          await this.sleep(step.delay || 30);
        }
        break;
      case "output":
        for (const line of step.lines) {
          if (!this.running) return;
          this.append("\n" + line);
          await this.sleep(50);
        }
        break;
      case "pause":
        await this.sleep(step.ms);
        break;
      case "clear":
        this.clear();
        break;
    }
  }

  append(html) {
    this.content += html;
    this.render();
  }

  render() {
    this.container.innerHTML = this.content + CURSOR;
    this.container.scrollTop = this.container.scrollHeight;
  }

  clear() {
    this.content = "";
    this.container.innerHTML = "";
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// --- Noise Overlay (ported from hermes-chat NoiseOverlay) ---
function initNoiseOverlay() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (typeof THREE === "undefined") return;

  const canvas = document.getElementById("noise-overlay");
  if (!canvas) return;

  const vertexShader = `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

  const fragmentShader = `
        uniform vec2 uRes;
        uniform float uDpr, uSize, uDensity, uOpacity;
        uniform vec3 uColor;
        varying vec2 vUv;

        float hash(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        void main() {
            float n = hash(floor(vUv * uRes / (uSize * uDpr)));
            gl_FragColor = vec4(uColor, step(1.0 - uDensity, n)) * uOpacity;
        }
    `;

  function hexToVec3(hex) {
    const c = hex.replace("#", "");
    return new THREE.Vector3(
      parseInt(c.substring(0, 2), 16) / 255,
      parseInt(c.substring(2, 4), 16) / 255,
      parseInt(c.substring(4, 6), 16) / 255,
    );
  }

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    canvas,
    premultipliedAlpha: false,
  });
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const geo = new THREE.PlaneGeometry(2, 2);

  const mat = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    transparent: true,
    uniforms: {
      uColor: { value: hexToVec3("#8090BB") },
      uDensity: { value: 0.1 },
      uDpr: { value: 1 },
      uOpacity: { value: 0.4 },
      uRes: { value: new THREE.Vector2() },
      uSize: { value: 1.0 },
    },
  });

  scene.add(new THREE.Mesh(geo, mat));

  function resize() {
    const dpr = window.devicePixelRatio;
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    renderer.setPixelRatio(dpr);
    mat.uniforms.uRes.value.set(w * dpr, h * dpr);
    mat.uniforms.uDpr.value = dpr;
  }

  resize();
  window.addEventListener("resize", resize);

  function loop() {
    requestAnimationFrame(loop);
    renderer.render(scene, camera);
  }
  loop();
}

// --- Initialize ---
document.addEventListener("DOMContentLoaded", () => {
  const detectedPlatform = detectPlatform();
  switchPlatform(detectedPlatform);

  initScrollAnimations();
  initNoiseOverlay();

  const terminalEl = document.getElementById("terminal-demo");

  if (terminalEl) {
    const demo = new TerminalDemo(terminalEl);

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            demo.start();
          } else {
            demo.stop();
          }
        });
      },
      { threshold: 0.3 },
    );

    observer.observe(document.querySelector(".terminal-window"));
  }

  const nav = document.querySelector(".nav");
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        if (window.scrollY > 50) {
          nav.style.borderBottomColor = "rgba(48, 80, 255, 0.15)";
        } else {
          nav.style.borderBottomColor = "";
        }
        ticking = false;
      });
      ticking = true;
    }
  });
});
