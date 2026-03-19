/**
 * X/Twitter Automation for Cosmos
 * Uses Puppeteer-Extra with Stealth Plugin
 * Debug mode (headless: false) for maximum stealth on Twitter
 */

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');

// Configure stealth plugin with all evasions
const stealth = StealthPlugin();

// Log available evasions for debugging
console.log('Available evasions:', stealth.availableEvasions);
console.log('Enabled evasions:', stealth.enabledEvasions);

// Apply stealth plugin
puppeteer.use(stealth);

const CONFIG = {
  // Load credentials from environment variables - NEVER HARDCODE!
  username: process.env.X_USERNAME || process.env.TWITTER_USERNAME || '',
  password: process.env.X_PASSWORD || process.env.TWITTER_PASSWORD || '',
  sessionDir: path.join(__dirname, 'session'),
  cookiesPath: path.join(__dirname, 'session', 'cookies.json'),
  localStoragePath: path.join(__dirname, 'session', 'localStorage.json'),
  userDataDir: path.join(__dirname, 'session', 'user_data'),

  // Debug mode - headless: false is MORE stealthy for Twitter
  // Twitter has aggressive detection that works better with visible browser
  headless: false,

  // Human-like delays
  slowMo: 30,

  // Viewport to match common resolution
  viewport: { width: 1366, height: 768 },
};

// Ensure session directory exists
if (!fs.existsSync(CONFIG.sessionDir)) {
  fs.mkdirSync(CONFIG.sessionDir, { recursive: true });
}

// Utility functions
async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function randomDelay(min = 800, max = 2500) {
  const delay = Math.floor(Math.random() * (max - min) + min);
  await sleep(delay);
}

// Human-like typing with mistakes and corrections
async function humanType(page, text) {
  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    // Occasionally make a typo and correct it (5% chance)
    if (Math.random() < 0.05 && i < text.length - 1) {
      const typo = String.fromCharCode(char.charCodeAt(0) + (Math.random() > 0.5 ? 1 : -1));
      await page.keyboard.type(typo, { delay: Math.random() * 100 + 50 });
      await sleep(Math.random() * 300 + 100);
      await page.keyboard.press('Backspace');
      await sleep(Math.random() * 200 + 50);
    }

    // Type the actual character with variable delay
    await page.keyboard.type(char, { delay: Math.random() * 150 + 30 });

    // Occasional pause (thinking)
    if (Math.random() < 0.08) {
      await sleep(Math.random() * 800 + 200);
    }
  }
}

// Save full session (cookies + localStorage)
async function saveSession(page) {
  try {
    // Save cookies
    const cookies = await page.cookies();
    fs.writeFileSync(CONFIG.cookiesPath, JSON.stringify(cookies, null, 2));

    // Save localStorage
    const localStorage = await page.evaluate(() => {
      const items = {};
      for (let i = 0; i < window.localStorage.length; i++) {
        const key = window.localStorage.key(i);
        items[key] = window.localStorage.getItem(key);
      }
      return items;
    });
    fs.writeFileSync(CONFIG.localStoragePath, JSON.stringify(localStorage, null, 2));

    console.log('✓ Session saved (cookies + localStorage)');
    return true;
  } catch (e) {
    console.error('Failed to save session:', e.message);
    return false;
  }
}

// Load full session
async function loadSession(page) {
  try {
    // Load cookies
    if (fs.existsSync(CONFIG.cookiesPath)) {
      const cookies = JSON.parse(fs.readFileSync(CONFIG.cookiesPath));
      await page.setCookie(...cookies);
      console.log('✓ Cookies loaded');
    }

    // Navigate first, then load localStorage
    await page.goto('https://x.com', { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Load localStorage
    if (fs.existsSync(CONFIG.localStoragePath)) {
      const localStorage = JSON.parse(fs.readFileSync(CONFIG.localStoragePath));
      await page.evaluate((items) => {
        for (const [key, value] of Object.entries(items)) {
          window.localStorage.setItem(key, value);
        }
      }, localStorage);
      console.log('✓ localStorage loaded');
    }

    return true;
  } catch (e) {
    console.error('Failed to load session:', e.message);
    return false;
  }
}

// Check if logged in
async function isLoggedIn(page) {
  try {
    await page.goto('https://x.com/home', { waitUntil: 'networkidle2', timeout: 45000 });
    await sleep(3000);

    const url = page.url();
    console.log('Current URL:', url);

    // Check for login redirects
    if (url.includes('/login') || url.includes('/i/flow/login')) {
      console.log('→ Not logged in (redirected to login)');
      return false;
    }

    // Look for home timeline indicators
    const homeIndicators = await page.$$('[data-testid="primaryColumn"]');
    if (homeIndicators.length > 0) {
      console.log('✓ Logged in (found home timeline)');
      return true;
    }

    // Alternative check - compose tweet button
    const composeBtn = await page.$('[data-testid="SideNav_NewTweet_Button"]');
    if (composeBtn) {
      console.log('✓ Logged in (found compose button)');
      return true;
    }

    console.log('→ Login status unclear');
    return false;
  } catch (e) {
    console.error('Login check error:', e.message);
    return false;
  }
}

// Perform login
async function login(page) {
  console.log('\n=== Starting Login Flow ===');

  // Set mobile user agent to try mobile site (often less protected)
  await page.setUserAgent('Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36');

  // Try mobile login flow which may have less bot detection
  await page.goto('https://mobile.twitter.com/login', { waitUntil: 'networkidle2', timeout: 60000 }).catch(async () => {
    // Fallback to regular x.com login
    await page.goto('https://twitter.com/i/flow/login', { waitUntil: 'networkidle2', timeout: 60000 });
  });
  await randomDelay(4000, 6000);

  // Take screenshot for debugging
  await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_step0.png') });
  console.log('Screenshot saved: login_step0.png');

  // Step 1: Enter username using more reliable method
  console.log('Step 1: Entering username...');
  try {
    // Wait for the username input to be ready and interactable
    await page.waitForSelector('input[autocomplete="username"]', { visible: true, timeout: 20000 });
    await randomDelay(1000, 2000);

    // Click and focus the input
    const usernameInput = await page.$('input[autocomplete="username"]');
    await usernameInput.click({ clickCount: 3 }); // Triple click to select all
    await randomDelay(300, 600);

    // Clear any existing text and type username
    await page.keyboard.press('Backspace');
    await randomDelay(200, 400);

    // Type username with human-like behavior
    await humanType(page, CONFIG.username);
    await randomDelay(1500, 2500);

    // Take screenshot after typing
    await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_step1.png') });
    console.log('Screenshot saved: login_step1.png');

    // Use keyboard Enter to submit (more human-like than clicking)
    console.log('Pressing Enter to submit username...');
    await page.keyboard.press('Enter');
    await randomDelay(4000, 6000);

  } catch (e) {
    console.error('Username entry failed:', e.message);
    await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_error1.png') });
    return false;
  }

  // Take screenshot after submitting username
  await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_step2.png') });
  console.log('Screenshot saved: login_step2.png');
  console.log('Current URL:', page.url());

  // Step 2: Check for unusual activity challenge (email/phone verification)
  await sleep(2000);
  let challengeInput = await page.$('input[data-testid="ocfEnterTextTextInput"]');
  if (challengeInput) {
    console.log('⚠ Unusual activity challenge detected - entering email/phone');
    await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_challenge.png') });

    await challengeInput.click();
    await randomDelay(500, 1000);
    // Enter username again as verification
    await humanType(page, CONFIG.username);
    await randomDelay(1000, 2000);
    await page.keyboard.press('Enter');
    await randomDelay(4000, 6000);
  }

  // Step 3: Wait for password field with multiple strategies
  console.log('Step 3: Looking for password field...');
  try {
    // First, let's see what's on the page
    const currentUrl = page.url();
    console.log('URL after username:', currentUrl);

    // Wait for navigation to complete
    await sleep(3000);
    await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_before_password.png') });

    // Try to find password field with various selectors
    const passwordSelectors = [
      'input[type="password"]',
      'input[name="password"]',
      'input[autocomplete="current-password"]',
      '[data-testid="LoginForm_Login_Button"]' // Sometimes this appears first
    ];

    let passwordField = null;
    for (const selector of passwordSelectors) {
      passwordField = await page.$(selector);
      if (passwordField) {
        console.log(`Found element with selector: ${selector}`);
        break;
      }
    }

    if (!passwordField) {
      console.log('Password field not found immediately, waiting longer...');
      await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_no_password.png') });

      // Try waiting for password field to appear
      try {
        await page.waitForSelector('input[type="password"]', { visible: true, timeout: 20000 });
        passwordField = await page.$('input[type="password"]');
      } catch (waitErr) {
        console.log('Still no password field after waiting');

        // Check if we're back on login page (bot detection)
        const pageContent = await page.content();
        if (pageContent.includes('Phone, email, or username')) {
          console.log('⚠ Page reset to initial login - possible bot detection');
          console.log('Trying alternative approach...');

          // Try direct navigation to password entry
          await page.goto(`https://x.com/i/flow/login?input_flow_data=%7B%22requested_variant%22%3A%22eyJsYW5nIjoiZW4ifQ%3D%3D%22%7D`, { waitUntil: 'networkidle2' });
          await randomDelay(3000, 5000);
        }
      }
    }

    if (passwordField && await passwordField.evaluate(el => el.type === 'password')) {
      console.log('Found password field, entering password...');
      await passwordField.click();
      await randomDelay(500, 1000);
      await humanType(page, CONFIG.password);
      await randomDelay(1500, 2500);

      // Take screenshot
      await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_step3.png') });

      // Submit with Enter key
      console.log('Pressing Enter to submit password...');
      await page.keyboard.press('Enter');
      await randomDelay(6000, 10000);
    } else {
      console.error('Could not find password field');
      await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_error_no_pw.png') });
      return false;
    }
  } catch (e) {
    console.error('Password entry failed:', e.message);
    await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_error3.png') });
    return false;
  }

  // Take final screenshot
  await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_final.png') });
  console.log('Final URL:', page.url());

  // Verify login success
  const loggedIn = await isLoggedIn(page);
  if (loggedIn) {
    await saveSession(page);
    console.log('=== Login Successful ===\n');
    return true;
  }

  console.log('=== Login May Have Failed ===\n');
  return false;
}

// Post a tweet
async function postTweet(page, content) {
  console.log('\n=== Posting Tweet ===');
  console.log('Content:', content.substring(0, 50) + '...');

  try {
    // Ensure we're on home
    await page.goto('https://x.com/home', { waitUntil: 'networkidle2', timeout: 30000 });
    await randomDelay(2000, 3000);

    // Method 1: Click compose button in sidebar
    let composeClicked = false;
    const composeBtn = await page.$('[data-testid="SideNav_NewTweet_Button"]');
    if (composeBtn) {
      await composeBtn.click();
      composeClicked = true;
      await randomDelay(1500, 2500);
    }

    // Method 2: Use the inline composer on home
    if (!composeClicked) {
      const inlineComposer = await page.$('[data-testid="tweetTextarea_0"]');
      if (inlineComposer) {
        await inlineComposer.click();
        await randomDelay(500, 1000);
      }
    }

    // Find and fill the tweet textarea
    const textarea = await page.waitForSelector('[data-testid="tweetTextarea_0"]', { timeout: 10000 });
    await textarea.click();
    await randomDelay(500, 1000);

    // Type the tweet with human-like behavior
    await humanType(page, content);
    await randomDelay(1500, 3000);

    // Click the Post button
    const postBtn = await page.$('[data-testid="tweetButton"]');
    if (postBtn) {
      // Check if button is enabled
      const isDisabled = await page.evaluate(el => el.hasAttribute('disabled'), postBtn);
      if (isDisabled) {
        console.log('⚠ Post button is disabled');
        return false;
      }

      await postBtn.click();
      console.log('✓ Post button clicked');
      await randomDelay(3000, 5000);

      // Save session after successful action
      await saveSession(page);

      console.log('=== Tweet Posted Successfully ===\n');
      return true;
    }

    console.log('✗ Could not find post button');
    return false;
  } catch (e) {
    console.error('Tweet posting error:', e.message);
    return false;
  }
}

// Create browser instance
async function createBrowser() {
  console.log('\n=== Launching Browser ===');
  console.log('Headless:', CONFIG.headless);
  console.log('User Data Dir:', CONFIG.userDataDir);

  const browser = await puppeteer.launch({
    headless: CONFIG.headless, // false = debug mode for better stealth
    userDataDir: CONFIG.userDataDir,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-blink-features=AutomationControlled',
      '--disable-infobars',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-background-networking',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-breakpad',
      '--disable-component-extensions-with-background-pages',
      '--disable-component-update',
      '--disable-default-apps',
      '--disable-extensions',
      '--disable-features=TranslateUI',
      '--disable-hang-monitor',
      '--disable-ipc-flooding-protection',
      '--disable-popup-blocking',
      '--disable-prompt-on-repost',
      '--disable-renderer-backgrounding',
      '--disable-sync',
      '--force-color-profile=srgb',
      '--metrics-recording-only',
      '--window-size=1366,768',
      '--lang=en-US,en',
    ],
    ignoreDefaultArgs: ['--enable-automation', '--enable-blink-features=IdleDetection'],
    slowMo: CONFIG.slowMo,
  });

  console.log('✓ Browser launched');
  return browser;
}

// Main function to post
async function post(tweetContent) {
  let browser = null;

  try {
    browser = await createBrowser();
    const page = await browser.newPage();

    // Set viewport
    await page.setViewport(CONFIG.viewport);

    // Set extra headers to appear more human
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
    });

    // Load existing session
    await loadSession(page);

    // Check login status
    let loggedIn = await isLoggedIn(page);

    // Login if needed
    if (!loggedIn) {
      loggedIn = await login(page);
      if (!loggedIn) {
        throw new Error('Failed to login to X');
      }
    }

    // Post the tweet
    const success = await postTweet(page, tweetContent);

    return success;
  } catch (error) {
    console.error('Error:', error.message);
    return false;
  } finally {
    if (browser) {
      // Keep browser open briefly to ensure actions complete
      await sleep(2000);
      await browser.close();
      console.log('Browser closed');
    }
  }
}

// Export for external use
module.exports = { post, createBrowser, login, postTweet, isLoggedIn, saveSession, loadSession };

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const content = args.join(' ') || `Good news everyone! 🧪

The Cosmos AI Swarm is autonomously building:
- Memory systems
- Context management
- Self-improving code

No human prompts needed!

👀 https://ai.cosmos.cloud
⭐ https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2

#AI #AutonomousAgents`;

  console.log('Tweet content:', content);
  console.log('');

  post(content)
    .then(success => {
      console.log('\nResult:', success ? 'SUCCESS' : 'FAILED');
      process.exit(success ? 0 : 1);
    })
    .catch(err => {
      console.error('Fatal error:', err);
      process.exit(1);
    });
}
