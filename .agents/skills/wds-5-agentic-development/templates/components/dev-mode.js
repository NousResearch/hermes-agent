/* eslint-disable n/no-unsupported-features/node-builtins */
/* global document, window */

/**
 * PROTOTYPE DEV MODE
 *
 * Developer/feedback mode that allows users to easily copy Object IDs to clipboard
 * for providing precise feedback on prototype elements.
 *
 * Features:
 * - Toggle dev mode with button or Ctrl+E
 * - Prototype works NORMALLY when dev mode is on
 * - Hold Shift + Click any element to copy its Object ID
 * - Visual highlights show what will be copied (green when Shift is held)
 * - Tooltip shows Object ID on hover
 * - Success feedback when copied
 *
 * Usage:
 * 1. Include this script in your prototype HTML
 * 2. Add the HTML toggle button and tooltip (see HTML template)
 * 3. Add the CSS styles (see CSS template)
 * 4. Call initDevMode() on page load
 *
 * How it works:
 * - Activate dev mode (Ctrl+E or click button)
 * - Hover over elements to see their Object IDs (gray outline)
 * - Hold Shift key (outline turns green)
 * - Click while holding Shift to copy Object ID
 * - Prototype works normally without Shift held
 * - **Shift is disabled when typing in form fields** (input, textarea, etc.)
 */

// ============================================================================
// DEV MODE STATE
// ============================================================================

let devModeActive = false;
let shiftKeyPressed = false;
let currentHighlightedElement = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

function initDevMode() {
  const toggleButton = document.querySelector('#dev-mode-toggle');
  const tooltip = document.querySelector('#dev-mode-tooltip');

  if (!toggleButton || !tooltip) {
    console.warn('⚠️ Dev Mode: Toggle button or tooltip not found');
    return;
  }

  // Check if user agent supports clipboard API
  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    // Clipboard API available
  } else {
    console.warn('⚠️ Clipboard API not supported in this browser');
    return;
  }

  setupKeyboardShortcuts();
  setupToggleButton(toggleButton, tooltip);
  setupHoverHighlight(tooltip);
  setupClickCopy();

  console.log('%c💡 Dev Mode available: Press Ctrl+E or click the Dev Mode button', 'color: #0066CC; font-weight: bold;');
}

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

function setupKeyboardShortcuts() {
  // Track Shift key for container selection
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') {
      // Don't activate if user is typing in a form field
      if (isTypingInField()) {
        return;
      }

      shiftKeyPressed = true;
      document.body.classList.add('shift-held');
      if (devModeActive) {
        console.log('%c⬆️ Shift held: Click any element to copy its Object ID', 'color: #10B981; font-weight: bold;');
      }
    }

    // Ctrl+E toggle
    if (e.ctrlKey && e.key === 'e') {
      e.preventDefault();
      document.querySelector('#dev-mode-toggle')?.click();
    }
  });

  document.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') {
      shiftKeyPressed = false;
      document.body.classList.remove('shift-held');
      if (devModeActive) {
        console.log('%c⬇️ Shift released: Prototype works normally (hold Shift to copy)', 'color: #6b7280;');
      }
    }
  });
}

// ============================================================================
// TOGGLE BUTTON
// ============================================================================

function setupToggleButton(toggleButton, tooltip) {
  toggleButton.addEventListener('click', function (e) {
    e.stopPropagation();
    if (typeof globalThis !== 'undefined') {
      globalThis.devModeActive = true;
    } else if (globalThis.window !== undefined) {
      globalThis.devModeActive = true;
    }
    devModeActive = !devModeActive;

    // Update UI
    document.body.classList.toggle('dev-mode-active', devModeActive);
    toggleButton.classList.toggle('active', devModeActive);

    const statusText = toggleButton.querySelector('span');
    if (statusText) {
      statusText.textContent = devModeActive ? 'Dev Mode: ON' : 'Dev Mode: OFF';
    }

    // Log status
    console.log(`🔧 Dev Mode: ${devModeActive ? 'ACTIVATED' : 'DEACTIVATED'}`);

    if (devModeActive) {
      console.log('%c🔧 DEV MODE ACTIVE', 'color: #0066CC; font-size: 16px; font-weight: bold;');
      console.log('%c⚠️ Hold SHIFT + Click any element to copy its Object ID', 'color: #FFB800; font-size: 14px; font-weight: bold;');
      console.log('%cWithout Shift: Prototype works normally', 'color: #6b7280;');
      console.log('%cPress Ctrl+E to toggle Dev Mode', 'color: #6b7280;');
    } else {
      tooltip.style.display = 'none';
      if (currentHighlightedElement) {
        clearHighlight();
      }
    }
  });
}

// ============================================================================
// HOVER HIGHLIGHT
// ============================================================================

function setupHoverHighlight(tooltip) {
  // Show tooltip and highlight on hover
  document.addEventListener('mouseover', function (e) {
    if (!devModeActive) return;

    // Don't highlight if user is typing in a field
    if (isTypingInField()) {
      tooltip.style.display = 'none';
      clearHighlight();
      return;
    }

    clearHighlight();

    let element = findElementWithId(e.target);

    if (!element || !element.id || isSystemElement(element.id)) {
      tooltip.style.display = 'none';
      return;
    }

    // Highlight element
    highlightElement(element, shiftKeyPressed);
    currentHighlightedElement = element;

    // Show tooltip
    const prefix = shiftKeyPressed ? '✓ Click to Copy: ' : '⬆️ Hold Shift + Click: ';
    tooltip.textContent = prefix + element.id;
    tooltip.style.display = 'block';
    tooltip.style.background = shiftKeyPressed ? '#10B981' : '#6b7280';
    tooltip.style.color = '#fff';

    updateTooltipPosition(e, tooltip);
  });

  // Update tooltip position on mouse move
  document.addEventListener('mousemove', function (e) {
    if (devModeActive && tooltip.style.display === 'block') {
      updateTooltipPosition(e, tooltip);
    }
  });

  // Clear highlight on mouse out
  document.addEventListener('mouseout', function (e) {
    if (!devModeActive) return;
    if (e.target.id) {
      tooltip.style.display = 'none';
      clearHighlight();
    }
  });
}

// ============================================================================
// CLICK TO COPY
// ============================================================================

function setupClickCopy() {
  // Use capture phase to intercept clicks with Shift
  document.addEventListener(
    'click',
    function (e) {
      if (!devModeActive) return;

      // Allow toggle button to work normally
      if (isToggleButton(e.target)) return;

      // ONLY copy if Shift is held
      if (!shiftKeyPressed) {
        // Let prototype work normally without Shift
        return;
      }

      // Don't intercept if user is clicking in/around a form field
      if (isTypingInField() || isFormElement(e.target)) {
        return;
      }

      // Shift is held and not in a form field - intercept and copy
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      let element = findElementWithId(e.target);

      if (!element || !element.id || isSystemElement(element.id)) {
        console.log('❌ No Object ID found');
        return false;
      }

      // Copy to clipboard
      const objectId = element.id;
      copyToClipboard(objectId);

      // Show feedback
      showCopyFeedback(element, objectId);

      return false;
    },
    true,
  ); // Capture phase
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function findElementWithId(element) {
  let current = element;
  let attempts = 0;

  while (current && !current.id && attempts < 10) {
    current = current.parentElement;
    attempts++;
  }

  return current;
}

function isSystemElement(id) {
  const systemIds = ['app', 'dev-mode-toggle', 'dev-mode-tooltip'];
  return systemIds.includes(id);
}

function isToggleButton(element) {
  return element.id === 'dev-mode-toggle' || element.closest('#dev-mode-toggle') || element.classList.contains('dev-mode-toggle');
}

function isTypingInField() {
  const activeElement = document.activeElement;
  if (!activeElement) return false;

  const tagName = activeElement.tagName.toLowerCase();
  const isEditable = activeElement.isContentEditable;

  // Check if user is currently typing in a form field
  return tagName === 'input' || tagName === 'textarea' || tagName === 'select' || isEditable;
}

function isFormElement(element) {
  if (!element) return false;

  const tagName = element.tagName.toLowerCase();
  const isEditable = element.isContentEditable;

  // Check if the clicked element is a form element
  return tagName === 'input' || tagName === 'textarea' || tagName === 'select' || isEditable;
}

function highlightElement(element, isShiftHeld) {
  const color = isShiftHeld ? '#10B981' : '#6b7280';
  const width = isShiftHeld ? '3px' : '2px';
  const offset = isShiftHeld ? '3px' : '2px';
  const shadowSpread = isShiftHeld ? '5px' : '2px';
  const shadowOpacity = isShiftHeld ? '0.4' : '0.2';

  element.style.outline = `${width} solid ${color}`;
  element.style.outlineOffset = offset;
  element.style.boxShadow = `0 0 0 ${shadowSpread} rgba(${isShiftHeld ? '16, 185, 129' : '107, 114, 128'}, ${shadowOpacity})`;
}

function clearHighlight() {
  if (currentHighlightedElement) {
    currentHighlightedElement.style.outline = '';
    currentHighlightedElement.style.boxShadow = '';
    currentHighlightedElement = null;
  }
}

function updateTooltipPosition(e, tooltip) {
  const offset = 15;
  let x = e.clientX + offset;
  let y = e.clientY + offset;

  // Keep tooltip on screen
  const rect = tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) {
    x = e.clientX - rect.width - offset;
  }
  if (y + rect.height > window.innerHeight) {
    y = e.clientY - rect.height - offset;
  }

  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}

function copyToClipboard(text) {
  if (typeof navigator !== 'undefined' && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        console.log(`📋 Copied to clipboard: ${text}`);
      })
      .catch((error) => {
        console.error('Dev Mode error:', error);
        fallbackCopy(text);
      });
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.left = '-999999px';
  document.body.append(textarea);
  textarea.focus();
  textarea.select();

  try {
    document.execCommand('copy');
    console.log(`📋 Copied (fallback): ${text}`);
  } catch (error) {
    console.error('Dev Mode error:', error);
  }

  textarea.remove();
}

function showCopyFeedback(element, objectId) {
  // Create feedback overlay
  const feedback = document.createElement('div');
  feedback.textContent = '✓ Copied: ' + objectId;
  feedback.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #10B981;
        color: #fff;
        padding: 16px 32px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        z-index: 100000;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        animation: fadeInOut 1.5s ease-in-out;
        pointer-events: none;
    `;

  document.body.append(feedback);

  setTimeout(() => {
    feedback.remove();
  }, 1500);

  // Flash element
  const originalOutline = element.style.outline;
  element.style.outline = '3px solid #10B981';
  setTimeout(() => {
    element.style.outline = originalOutline;
  }, 300);
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
        20% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        80% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
    }
`;
document.head.append(style);

// ============================================================================
// EXPORT
// ============================================================================

// Make available globally
globalThis.initDevMode = initDevMode;

// Export for use in other scripts
if (typeof globalThis !== 'undefined' && globalThis.exports) {
  globalThis.exports = { initDevMode };
}
