
# ============================================
# CLASS 5: LYAPUNOV STABILITY CORE
# ============================================

def calculate_text_phase(text: str) -> float:
    """
    Estimate geometric phase (0 to π/2) from text content.
    Used to check if AI response aligns with user's emotional phase.
    
    Heuristic Mapping:
    - Short/Guarded/Analysis -> Low Phase (Masking)
    - Balanced/Warm/Collaborative -> Mid Phase (Synchrony)
    - Chaotic/Caps/Exclamations -> High Phase (Leakage)
    """
    if not text:
        return PHASE_SYNCHRONY
        
    # 1. Feature Extraction
    length = len(text)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(length, 1)
    exclamations = text.count('!')
    questions = text.count('?')
    
    # 2. Base Phase (Start at Synchrony)
    phase = PHASE_SYNCHRONY
    
    # 3. Modulation
    # High Caps/Exclamations -> Push towards Leakage (High Phase)
    if caps_ratio > 0.2 or exclamations > 1:
        phase += 0.5  # Push towards 1.2+
        
    # Short/Terse -> Push towards Masking (Low Phase)
    if length < 50 and questions == 0:
        phase -= 0.4  # Push towards 0.3
        
    # Analytical words -> Push towards Low Phase
    analytical_words = ['analysis', 'logic', 'data', 'verify', 'incorrect']
    if any(keyword in text.lower() for keyword in analytical_words):
        phase -= 0.2
        
    # Emotional words -> Push towards High Phase
    emotional_words = ['feel', 'love', 'hate', 'scared', 'amazing', 'wow']
    if any(keyword in text.lower() for keyword in emotional_words):
        phase += 0.2
        
    # Clamp to valid range [0, π/2]
    return max(0.0, min(math.pi / 2, phase))


def check_lyapunov_stability(user_phase: float, response_text: str, 
                             threshold: float = 0.15) -> Tuple[bool, float, float]:
    """
    Check if the AI response is stable relative to the user's phase.
    
    Returns:
        (is_stable, drift, response_phase)
    """
    # Calculate phase of the proposed response
    response_phase = calculate_text_phase(response_text)
    
    # Calculate drift (Lyapunov Exponent proxy)
    drift = abs(user_phase - response_phase)
    
    # Check stability
    is_stable = drift <= threshold
    
    return is_stable, drift, response_phase
