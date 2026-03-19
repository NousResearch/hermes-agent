/**
 * Cosmos 54D CNS: The Symbiotic Mirror
 * Standalone Showcase Logic
 */

function initShowcase() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    let skullPoints = [];
    let animationId;
    let targetBPM = 72;
    let currentBPM = 72;
    let entropyValue = 0.842;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        generateSkullPoints();
    }

    function generateSkullPoints() {
        skullPoints = [];
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        // Responsive scaling
        const scale = Math.min(canvas.width, canvas.height) * 0.28;

        // Cranium
        for (let angle = 0; angle < Math.PI; angle += 0.04) {
            skullPoints.push({
                x: centerX + Math.cos(angle - Math.PI) * scale,
                y: centerY + Math.sin(angle - Math.PI) * scale * 1.15
            });
        }

        // Jaw
        for (let x = -0.55; x <= 0.55; x += 0.04) {
            const y = 0.95 + Math.pow(x, 2) * 0.45;
            skullPoints.push({
                x: centerX + x * scale,
                y: centerY + y * scale
            });
        }

        // Eye Sockets
        for (let angle = 0; angle < Math.PI * 2; angle += 0.15) {
            skullPoints.push({ x: centerX - scale * 0.32 + Math.cos(angle) * scale * 0.12, y: centerY - scale * 0.05 + Math.sin(angle) * scale * 0.12 });
            skullPoints.push({ x: centerX + scale * 0.32 + Math.cos(angle) * scale * 0.12, y: centerY - scale * 0.05 + Math.sin(angle) * scale * 0.12 });
        }

        // Nasal Cavity
        skullPoints.push({ x: centerX, y: centerY + scale * 0.25 });
        skullPoints.push({ x: centerX - scale * 0.08, y: centerY + scale * 0.4 });
        skullPoints.push({ x: centerX + scale * 0.08, y: centerY + scale * 0.4 });
    }

    function createParticles() {
        particles = [];
        const count = 1200; // Dense enough for "12D" feel

        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                size: Math.random() * 1.5 + 0.5,
                target: skullPoints[i % skullPoints.length],
                color: Math.random() > 0.9 ? '#06b6d4' : '#8b5cf6',
                alpha: Math.random() * 0.5 + 0.2
            });
        }
    }

    function draw() {
        // Clear with slight trail
        ctx.fillStyle = 'rgba(3, 5, 8, 0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const time = Date.now() * 0.001;
        // Pulse logic synced with BPM
        const pulseRatio = (currentBPM / 60);
        const pulse = Math.sin(time * pulseRatio * Math.PI) * 0.06 + 1.0;

        particles.forEach((p, i) => {
            // Target scaling with pulse
            const tx = p.target.x + (p.target.x - canvas.width / 2) * (pulse - 1);
            const ty = p.target.y + (p.target.y - canvas.height / 2) * (pulse - 1);

            const dx = tx - p.x;
            const dy = ty - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Accelerator physics
            p.vx += dx * 0.004;
            p.vy += dy * 0.004;

            // Brownian motion (Cognitive Entropy)
            p.vx += (Math.random() - 0.5) * (entropyValue * 0.5);
            p.vy += (Math.random() - 0.5) * (entropyValue * 0.5);

            // Friction
            p.vx *= 0.92;
            p.vy *= 0.92;

            p.x += p.vx;
            p.y += p.vy;

            // Render
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.alpha * (0.5 + 0.5 * pulse);
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * (0.8 + 0.4 * pulse), 0, Math.PI * 2);
            ctx.fill();

            // Connections (Sparse neural net)
            if (i % 25 === 0) {
                const next = particles[(i + 1) % particles.length];
                const d = Math.sqrt(Math.pow(p.x - next.x, 2) + Math.pow(p.y - next.y, 2));
                if (d < 100) {
                    ctx.strokeStyle = '#8b5cf6';
                    ctx.globalAlpha = 0.05 * (1 - d / 100);
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(next.x, next.y);
                    ctx.stroke();
                }
            }
        });

        // Loop
        animationId = requestAnimationFrame(draw);
    }

    // Data Mocking Sync
    function simulateData() {
        // Subtle drift in entropy and BPM
        entropyValue = 0.84 + Math.sin(Date.now() / 3000) * 0.01;
        document.getElementById('entropy-value').textContent = entropyValue.toFixed(3);

        // Smooth BPM transition
        if (Math.random() > 0.99) targetBPM = 65 + Math.random() * 15;
        currentBPM += (targetBPM - currentBPM) * 0.01;

        requestAnimationFrame(simulateData);
    }

    // Scroll Interactivity
    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;
        const heroHeight = window.innerHeight;
        // Shift skull out of view as we scroll
        if (scrolled < heroHeight) {
            // Shift points down or fade?
        }
    });

    // Quantum Log Logic
    const quantumLogs = [
        "Initializing 5-Qubit Geometric Entanglement...",
        "Bridge Sync: ΦG 43.19° | Entropy: 0.841",
        "Injecting User Bio-Metric Pulse: 72 BPM",
        "Executing Hadamard Transform on Q0, Q2...",
        "Measuring State Displacement... Delta < 0.05",
        "Waveform Collapse: |10101⟩ (Resonance: 0.92)",
        "54D CNS Weight Adjustment: Plasticity Update +0.02",
        "Swarm Consensus Reached: Hermes-4 valid",
        "Entropy Threshold Maintained (0.842)"
    ];

    function startLogs() {
        const logContainer = document.createElement('div');
        logContainer.id = 'quantum-logs';
        document.getElementById('quantum').querySelector('.section-container').appendChild(logContainer);
        
        let i = 0;
        setInterval(() => {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="time">[${new Date().toLocaleTimeString()}]</span> ${quantumLogs[i % quantumLogs.length]}`;
            logContainer.prepend(logEntry);
            if (logContainer.children.length > 8) logContainer.lastChild.remove();
            i++;
        }, 1500);
    }

    resize();
    createParticles();
    draw();
    simulateData();
    startLogs();

    window.addEventListener('resize', resize);
}

// Glitch Effect Helper
function initGlitch() {
    const glitchElements = document.querySelectorAll('.glitch-text');
    glitchElements.forEach(el => {
        setInterval(() => {
            if (Math.random() > 0.95) {
                el.style.transform = `translate(${(Math.random()-0.5)*10}px, ${(Math.random()-0.5)*5}px)`;
                el.style.color = Math.random() > 0.5 ? '#06b6d4' : '#ec4899';
                setTimeout(() => {
                    el.style.transform = 'none';
                    el.style.color = 'white';
                }, 100);
            }
        }, 200);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initShowcase();
    initGlitch();
});
