// =========================================================================
// Hermes Agent Landing Page — Interactions
// =========================================================================

// --- Copy to clipboard ---
function copyInstall() {
    const text = document.getElementById('install-command').textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.hero-install .copy-btn');
        const original = btn.querySelector('.copy-text').textContent;
        btn.querySelector('.copy-text').textContent = 'Copied!';
        btn.style.color = 'var(--gold)';
        setTimeout(() => {
            btn.querySelector('.copy-text').textContent = original;
            btn.style.color = '';
        }, 2000);
    });
}

function copyText(btn) {
    const text = btn.getAttribute('data-text');
    navigator.clipboard.writeText(text).then(() => {
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        btn.style.color = 'var(--gold)';
        setTimeout(() => {
            btn.textContent = original;
            btn.style.color = '';
        }, 2000);
    });
}

// --- Scroll-triggered fade-in ---
function initScrollAnimations() {
    const elements = document.querySelectorAll(
        '.feature-card, .tool-pill, .platform-group, .skill-category, ' +
        '.install-step, .research-card, .footer-card, .section-header, ' +
        '.lead-text, .section-desc, .terminal-window'
    );

    elements.forEach(el => el.classList.add('fade-in'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Stagger children within grids
                const parent = entry.target.parentElement;
                if (parent) {
                    const siblings = parent.querySelectorAll('.fade-in');
                    let idx = Array.from(siblings).indexOf(entry.target);
                    if (idx < 0) idx = 0;
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                    }, idx * 60);
                } else {
                    entry.target.classList.add('visible');
                }
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    elements.forEach(el => observer.observe(el));
}

// --- Terminal Demo ---
const demoSequence = [
    // Scene 1: Research task with delegation
    { type: 'prompt', text: '❯ ' },
    { type: 'type', text: 'Research the latest approaches to GRPO training and write a summary', delay: 30 },
    { type: 'pause', ms: 600 },
    { type: 'output', lines: [
        '',
        '<span class="t-dim">┊ 🔍 web_search "GRPO reinforcement learning 2026"      1.2s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 📄 web_extract arxiv.org/abs/2402.03300               3.1s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 🔍 web_search "GRPO vs PPO ablation results"          0.9s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 📄 web_extract huggingface.co/blog/grpo               2.8s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ ✍️  write_file ~/research/grpo-summary.md               0.1s</span>',
    ]},
    { type: 'pause', ms: 500 },
    { type: 'output', lines: [
        '',
        '<span class="t-text">Done! I\'ve written a summary covering:</span>',
        '',
        '<span class="t-text">  <span class="t-green">✓</span> GRPO\'s group-relative advantage (no critic model needed)</span>',
        '<span class="t-text">  <span class="t-green">✓</span> Comparison with PPO/DPO on reasoning benchmarks</span>',
        '<span class="t-text">  <span class="t-green">✓</span> Implementation notes for Axolotl and TRL</span>',
        '',
        '<span class="t-text">Saved to</span> <span class="t-amber">~/research/grpo-summary.md</span>',
    ]},
    { type: 'pause', ms: 2500 },

    // Scene 2: Quick delegation
    { type: 'clear' },
    { type: 'prompt', text: '❯ ' },
    { type: 'type', text: 'Review the PR at NousResearch/hermes-agent#42 and fix any issues', delay: 30 },
    { type: 'pause', ms: 600 },
    { type: 'output', lines: [
        '',
        '<span class="t-dim">┊ 🔀 delegate_task "review PR #42 changes"               2.1s</span>',
    ]},
    { type: 'pause', ms: 500 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 💻 git diff main..pr-42                                0.4s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ ✏️  patch tools/registry.py                             0.1s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 💻 python -m pytest tests/ -x                          3.2s</span>',
    ]},
    { type: 'pause', ms: 400 },
    { type: 'output', lines: [
        '<span class="t-dim">┊ 💻 git commit -m "fix: handle empty tool schemas"      0.3s</span>',
    ]},
    { type: 'pause', ms: 500 },
    { type: 'output', lines: [
        '',
        '<span class="t-text">Found 2 issues in the PR and fixed both:</span>',
        '',
        '<span class="t-text">  <span class="t-green">✓</span> Empty tool schema crash in registry.py — added guard</span>',
        '<span class="t-text">  <span class="t-green">✓</span> Missing error handling in delegate_tool.py — added try/except</span>',
        '',
        '<span class="t-text">Tests pass. Committed the fix and pushed to the PR branch.</span>',
        '<span class="t-text">I also saved a</span> <span class="t-amber">skill</span> <span class="t-text">for this PR review pattern.</span>',
    ]},
    { type: 'pause', ms: 2500 },

    // Scene 3: Session search by keyword
    { type: 'clear' },
    { type: 'prompt', text: '❯ ' },
    { type: 'type', text: 'How did we fix that Docker networking issue?', delay: 35 },
    { type: 'pause', ms: 500 },
    { type: 'output', lines: [
        '',
        '<span class="t-dim">┊ 🔎 session_search "Docker networking"                  1.4s</span>',
    ]},
    { type: 'pause', ms: 500 },
    { type: 'output', lines: [
        '',
        '<span class="t-text">Found it — from a session on February 12th:</span>',
        '',
        '<span class="t-text">The containers couldn\'t reach each other because the compose</span>',
        '<span class="t-text">file was using the default bridge network. We switched to a</span>',
        '<span class="t-text">custom network with</span> <span class="t-amber">driver: overlay</span><span class="t-text">, added explicit</span>',
        '<span class="t-text">aliases, and set</span> <span class="t-amber">dns: 8.8.8.8</span> <span class="t-text">as a fallback.</span>',
        '',
        '<span class="t-text">The fix was committed in</span> <span class="t-amber">docker-compose.prod.yml</span><span class="t-text">.</span>',
    ]},
    { type: 'pause', ms: 3000 },
];

class TerminalDemo {
    constructor(element, cursorElement) {
        this.el = element;
        this.cursor = cursorElement;
        this.running = false;
        this.content = '';
        this.observer = null;
    }

    async start() {
        if (this.running) return;
        this.running = true;
        
        while (this.running) {
            for (const step of demoSequence) {
                if (!this.running) return;
                await this.execute(step);
            }
            // Loop
            this.clear();
            await this.sleep(1000);
        }
    }

    stop() {
        this.running = false;
    }

    async execute(step) {
        switch (step.type) {
            case 'prompt':
                this.append(`<span class="t-prompt">${step.text}</span>`);
                break;

            case 'type':
                for (const char of step.text) {
                    if (!this.running) return;
                    this.append(`<span class="t-cmd">${char}</span>`);
                    await this.sleep(step.delay || 30);
                }
                break;

            case 'output':
                for (const line of step.lines) {
                    if (!this.running) return;
                    this.append('\n' + line);
                    await this.sleep(50);
                }
                break;

            case 'pause':
                await this.sleep(step.ms);
                break;

            case 'clear':
                this.clear();
                break;
        }
    }

    append(html) {
        this.content += html;
        this.el.innerHTML = this.content;
        // Keep cursor at end
        this.el.parentElement.scrollTop = this.el.parentElement.scrollHeight;
    }

    clear() {
        this.content = '';
        this.el.innerHTML = '';
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// --- Initialize ---
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();

    // Terminal demo - start when visible
    const terminalEl = document.getElementById('terminal-content');
    const cursorEl = document.getElementById('terminal-cursor');
    
    if (terminalEl && cursorEl) {
        const demo = new TerminalDemo(terminalEl, cursorEl);
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    demo.start();
                } else {
                    demo.stop();
                }
            });
        }, { threshold: 0.3 });

        observer.observe(document.querySelector('.terminal-window'));
    }

    // Smooth nav background on scroll
    const nav = document.querySelector('.nav');
    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                if (window.scrollY > 50) {
                    nav.style.borderBottomColor = 'rgba(255, 215, 0, 0.1)';
                } else {
                    nav.style.borderBottomColor = '';
                }
                ticking = false;
            });
            ticking = true;
        }
    });
});
