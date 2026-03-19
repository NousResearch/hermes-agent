
// ============================================
// CLASS 5: EMOTIONAL SENSORY BRIDGE
// ============================================

let emotionalWs = null;
let emotionalConnected = false;
let reconnectTimer = null;
let emotionalReconnectDelay = 3000; // Exponential backoff start

document.addEventListener('DOMContentLoaded', () => {
    // Start the bridge after a slight delay to ensure DOM is ready
    setTimeout(initEmotionalBridge, 1000);
});

function initEmotionalBridge() {
    console.log('[Symbiote] Initializing 12D CST Sensory Bridge...');
    connectEmotionalAPI();
}

function connectEmotionalAPI() {
    const statusText = document.getElementById('sensor-status-text');
    const statusIcon = document.getElementById('sensor-status-icon');
    const liveIndicator = document.getElementById('emotional-live-indicator');

    if (statusText) statusText.textContent = 'Connecting...';

    // Connect to Emotional API Port (8765)
    const wsUrl = 'ws://localhost:8765/ws';

    try {
        emotionalWs = new WebSocket(wsUrl);

        emotionalWs.onopen = () => {
            console.log('[Symbiote] Connected to Emotional API');
            emotionalConnected = true;
            emotionalReconnectDelay = 3000; // Reset backoff on success
            if (statusText) statusText.textContent = 'Connected';
            if (statusIcon) statusIcon.textContent = '✅';
            if (liveIndicator) liveIndicator.classList.add('active');

            if (reconnectTimer) clearTimeout(reconnectTimer);
        };

        emotionalWs.onmessage = (event) => {
            try {
                const packet = JSON.parse(event.data);
                if (packet.type === 'cosmos_packet') {
                    // Support both server (cosmos_packet) and demo (data) structures
                    const payload = packet.data || packet.cosmos_packet;
                    updateSymbioteUI(payload);
                }
            } catch (e) {
                console.error('Packet parse error:', e);
            }
        };

        emotionalWs.onclose = () => {
            console.log('[Symbiote] Disconnected');
            emotionalConnected = false;
            if (statusText) statusText.textContent = 'Disconnected';
            if (statusIcon) statusIcon.textContent = '❌';
            if (liveIndicator) liveIndicator.classList.remove('active');

            // Reconnect with exponential backoff (3s → 6s → 12s → max 30s)
            console.log(`[Symbiote] Reconnecting in ${emotionalReconnectDelay / 1000}s...`);
            reconnectTimer = setTimeout(connectEmotionalAPI, emotionalReconnectDelay);
            emotionalReconnectDelay = Math.min(emotionalReconnectDelay * 2, 30000);
        };

    } catch (e) {
        console.error('Emotional Bridge Error:', e);
    }
}

function updateSymbioteUI(data) {
    if (!data) return;

    // 1. CST Phase
    const phaseRad = data.cst_physics?.geometric_phase_rad || 0;
    const phaseDeg = Math.round(phaseRad * (180 / Math.PI));
    const phaseEl = document.getElementById('cst-phi-value');
    if (phaseEl) phaseEl.textContent = `ΦG: ${phaseDeg}°`;

    // 2. Class 5: Lyapunov Lock
    const entanglement = data.cst_physics?.quantum_entanglement || 0;
    const drift = Math.max(0, 1.0 - entanglement);

    const needle = document.getElementById('lyapunov-needle');
    const status = document.getElementById('lyapunov-status');
    const driftVal = document.getElementById('lyapunov-drift');

    if (needle) {
        const jitter = (Math.random() - 0.5) * 5;
        const deg = (drift * 90) - 45 + jitter;
        needle.style.transform = `rotate(${deg}deg)`;

        if (drift > 0.3) {
            needle.style.background = 'var(--error)';
            if (status) { status.textContent = 'UNSTABLE'; status.style.color = 'var(--error)'; }
        } else {
            needle.style.background = 'var(--success)';
            if (status) { status.textContent = 'LOCKED'; status.style.color = 'var(--success)'; }
        }
    }
    if (driftVal) driftVal.textContent = `${drift.toFixed(3)} rad`;

    // 3. Class 5: Emeth Harmonizer Stats
    // Use correct paths: derived_state.informational_mass, spectral_physics.spectral_flatness
    const mass = data.derived_state?.informational_mass || 0;
    const valence = data.derived_state?.pad_vector?.pleasure || 0;
    const flatness = data.spectral_physics?.spectral_flatness || 0;

    const percW = Math.min(100, mass * 2);  // mass is 0-100 scale
    const stringW = Math.min(100, ((valence + 1) / 2) * 100);
    const brassW = Math.min(100, flatness * 200);

    const percBar = document.getElementById('perc-bar');
    if (percBar) percBar.style.width = `${percW}%`;
    const stringBar = document.getElementById('string-bar');
    if (stringBar) stringBar.style.width = `${stringW}%`;
    const brassBar = document.getElementById('brass-bar');
    if (brassBar) brassBar.style.width = `${brassW}%`;

    // Conductor Instruction
    const instrEl = document.getElementById('conductor-instruction');
    if (instrEl) {
        if (percW > 70) instrEl.textContent = '"Boost Percussion (Logic) for High Mass"';
        else if (stringW > 70) instrEl.textContent = '"Swell Strings (Empathy) for High Valence"';
        else if (brassW > 60) instrEl.textContent = '"Amplify Brass (Creativity) for Entropy"';
        else instrEl.textContent = '"Maintain Balanced Orchestral Mix"';
    }

    // 4. Standard Metrics - Read from derived_state.pad_vector
    const arousal = data.derived_state?.pad_vector?.arousal || 0;
    const valenceVal = data.derived_state?.pad_vector?.pleasure || 0;

    const valenceBar = document.getElementById('valence-bar');
    if (valenceBar) valenceBar.style.width = `${((valenceVal + 1) / 2) * 100}%`;
    const valenceValue = document.getElementById('valence-value');
    if (valenceValue) valenceValue.textContent = valenceVal.toFixed(2);

    const arousalBar = document.getElementById('arousal-bar');
    if (arousalBar) arousalBar.style.width = `${Math.abs(arousal) * 100}%`;
    const arousalValue = document.getElementById('arousal-value');
    if (arousalValue) arousalValue.textContent = arousal.toFixed(2);

    // Intensity from informational_mass (0-100 scaled to 0-1)
    const intensity = Math.min(1.0, mass / 50);

    const intensityBar = document.getElementById('intensity-bar');
    if (intensityBar) intensityBar.style.width = `${intensity * 100}%`;
    const intensityValue = document.getElementById('intensity-value');
    if (intensityValue) intensityValue.textContent = intensity.toFixed(2);

    // Biometrics - Read from cst_physics.virtual_body
    const entropy = data.spectral_physics?.spectral_flatness || 0.5;

    // 4b. QUANTUM ENTROPY (The Real Deal)
    // If we have direct quantum bridge data, override the bio-entropy or add a new metric
    const quantumEntropy = data.cst_physics?.quantum_entropy || entropy;

    const bioEntropy = document.getElementById('bio-entropy');
    if (bioEntropy) {
        bioEntropy.textContent = quantumEntropy.toFixed(4);
        // Visual Flair: Color code based on chaos
        if (quantumEntropy > 0.8) bioEntropy.style.color = 'var(--accent)'; // High Chaos
        else if (quantumEntropy < 0.2) bioEntropy.style.color = 'var(--success)'; // Order
        else bioEntropy.style.color = 'var(--text-primary)';
    }

    // Heart/Breath - cst_physics.virtual_body for virtual mode
    const vBody = data.cst_physics?.virtual_body || {};
    const heartVal = vBody.heart_rate || '--';
    const breathVal = vBody.respiration_rate || '--';

    const bioHeart = document.getElementById('bio-heart');
    if (bioHeart) {
        bioHeart.textContent = typeof heartVal === 'number' ? `${heartVal.toFixed(0)} BPM` : heartVal;

        // Apply pulse animation to the icon
        const bioIcon = bioHeart.closest('.bio-item')?.querySelector('.bio-icon');
        if (bioIcon && typeof heartVal === 'number' && heartVal > 0) {
            bioIcon.classList.add('pulse-animation');
            const duration = 60 / heartVal; // 60s / BPM = seconds per beat
            bioIcon.style.animationDuration = `${duration}s`;
            
            // Sync neural canvas skull pulse
            if (window.neuralCanvasUpdateBPM) {
                window.neuralCanvasUpdateBPM(heartVal);
            }
        } else if (bioIcon) {
            bioIcon.classList.remove('pulse-animation');
        }
    }

    const bioBreath = document.getElementById('bio-breath');
    if (bioBreath) bioBreath.textContent = typeof breathVal === 'number' ? `${breathVal.toFixed(0)} BPM` : breathVal;


    const phaseEl2 = document.getElementById('cst-phase');
    // Use primary_affect_label from derived_state
    const primaryEmotion = data.derived_state?.primary_affect_label || 'CALIBRATING';
    if (phaseEl2) phaseEl2.textContent = primaryEmotion;

    // 5. Detected Emotions List - Read from derived_state.emotion_vectors
    const emotionsDiv = document.getElementById('detected-emotions');
    const vectors = data.derived_state?.emotion_vectors || data.emotional_state?.emotion_vectors;

    if (emotionsDiv && vectors) {
        // Show ALL emotions with non-zero value, sorted by intensity
        const sorted = Object.entries(vectors)
            .sort(([, a], [, b]) => b - a)
            .filter(([, val]) => val > 0.05); // Filter low noise

        emotionsDiv.innerHTML = sorted.map(([emo, val]) => `
            <span class="emotion-tag" data-emotion="${emo}" style="opacity: ${Math.max(0.6, val)};">
                ${getEmotionEmoji(emo)} ${emo} ${(val * 100).toFixed(0)}%
            </span>
        `).join('');

        if (sorted.length === 0 && primaryEmotion) {
            emotionsDiv.innerHTML = `
            <span class="emotion-tag" data-emotion="${primaryEmotion}">
                ${getEmotionEmoji(primaryEmotion)} ${primaryEmotion}
            </span>`;
        }
    }
}

function getEmotionEmoji(emo) {
    const map = {
        // Primary
        'JOY': '😊', 'TRUST': '🤝', 'FEAR': '😨', 'SURPRISE': '😲',
        'SADNESS': '😢', 'DISGUST': '🤢', 'ANGER': '😠', 'ANTICIPATION': '🎯',
        // Secondary
        'LOVE': '❤️', 'SUBMISSION': '🙇', 'AWE': '🤯', 'DISAPPROVAL': '👎',
        'REMORSE': '😔', 'CONTEMPT': '😏', 'AGGRESSIVENESS': '💢', 'OPTIMISM': '🌟',
        // Tertiary
        'SERENITY': '😌', 'ACCEPTANCE': '🤗', 'APPREHENSION': '😟', 'DISTRACTION': '🤔',
        'PENSIVENESS': '💭', 'BOREDOM': '😑', 'ANNOYANCE': '😤', 'INTEREST': '👀',
        // Virtual
        'NEUTRAL': '😐', 'CALIBRATING': '⏳', 'VIRTUAL_DREAMING': '💭',
        // Legacy
        'HAPPY': '😊', 'SAD': '😢', 'CALM': '😌', 'ANGRY': '😠'
    };
    return map[emo.toUpperCase()] || '😐';
}

// Define reconnect function and map globally
function reconnectEmotionalAPI() {
    emotionalReconnectDelay = 3000; // Reset backoff for manual reconnect
    connectEmotionalAPI();
}
window.reconnectEmotionalAPI = reconnectEmotionalAPI;
