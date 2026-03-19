/**
 * X/Twitter Manual Login Helper
 * Opens a browser for the user to manually login, then saves the session
 * This bypasses bot detection since a human is doing the actual login
 *
 * Usage: node x_login_helper.js
 * Then manually login in the browser that opens
 */

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Configure stealth
const stealth = StealthPlugin();
puppeteer.use(stealth);

const CONFIG = {
  sessionDir: path.join(__dirname, 'session'),
  cookiesPath: path.join(__dirname, 'session', 'cookies.json'),
  localStoragePath: path.join(__dirname, 'session', 'localStorage.json'),
  userDataDir: path.join(__dirname, 'session', 'user_data'),
  viewport: { width: 1366, height: 768 },
};

// Ensure session directory exists
if (!fs.existsSync(CONFIG.sessionDir)) {
  fs.mkdirSync(CONFIG.sessionDir, { recursive: true });
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function saveSession(page) {
  try {
    // Save cookies
    const cookies = await page.cookies();
    fs.writeFileSync(CONFIG.cookiesPath, JSON.stringify(cookies, null, 2));
    console.log(`✓ Saved ${cookies.length} cookies`);

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
    console.log(`✓ Saved ${Object.keys(localStorage).length} localStorage items`);

    return true;
  } catch (e) {
    console.error('Failed to save session:', e.message);
    return false;
  }
}

async function isLoggedIn(page) {
  try {
    const url = page.url();

    // Check for login page indicators
    if (url.includes('/login') || url.includes('/i/flow/login')) {
      return false;
    }

    // Check for home page or profile indicators
    if (url.includes('/home')) {
      const composeBtn = await page.$('[data-testid="SideNav_NewTweet_Button"]');
      if (composeBtn) return true;

      const timeline = await page.$('[data-testid="primaryColumn"]');
      if (timeline) return true;
    }

    return false;
  } catch (e) {
    return false;
  }
}

function askQuestion(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise(resolve => {
    rl.question(question, answer => {
      rl.close();
      resolve(answer);
    });
  });
}

async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('     X/TWITTER MANUAL LOGIN HELPER');
  console.log('='.repeat(60));
  console.log('\nThis will open a browser for you to manually login to X.');
  console.log('After logging in, the session will be saved for automated posting.\n');

  let browser = null;

  try {
    console.log('Launching browser...');
    browser = await puppeteer.launch({
      headless: false, // Must be visible for manual login
      userDataDir: CONFIG.userDataDir,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-blink-features=AutomationControlled',
        '--disable-infobars',
        '--window-size=1366,768',
        '--lang=en-US,en',
      ],
      ignoreDefaultArgs: ['--enable-automation'],
      defaultViewport: CONFIG.viewport,
    });

    const page = await browser.newPage();
    await page.setViewport(CONFIG.viewport);

    // Navigate to X login
    console.log('Navigating to X login page...\n');
    await page.goto('https://x.com/i/flow/login', { waitUntil: 'networkidle2', timeout: 60000 });

    console.log('='.repeat(60));
    console.log('BROWSER IS NOW OPEN');
    console.log('='.repeat(60));
    console.log('\nPlease complete the login in the browser window.');
    console.log('The login page should be visible now.');
    console.log('\nPlease login with your X/Twitter credentials.');
    console.log('(Set X_USERNAME and X_PASSWORD environment variables for future reference)');
    console.log('\nAfter logging in successfully:');
    console.log('1. Make sure you see the home timeline');
    console.log('2. Come back here and press ENTER\n');

    // Wait for user to press enter
    await askQuestion('Press ENTER once you have logged in successfully... ');

    console.log('\nChecking login status...');

    // Navigate to home to verify
    await page.goto('https://x.com/home', { waitUntil: 'networkidle2', timeout: 30000 });
    await sleep(3000);

    const loggedIn = await isLoggedIn(page);

    if (loggedIn) {
      console.log('\n✓ Login confirmed!');
      console.log('Saving session...');
      await saveSession(page);

      // Take a screenshot as proof
      await page.screenshot({ path: path.join(CONFIG.sessionDir, 'login_success.png') });
      console.log('✓ Screenshot saved: login_success.png');

      console.log('\n' + '='.repeat(60));
      console.log('SESSION SAVED SUCCESSFULLY!');
      console.log('='.repeat(60));
      console.log('\nYou can now close this window.');
      console.log('Future automated posts will use this saved session.\n');
    } else {
      console.log('\n✗ Could not verify login.');
      console.log('Please make sure you:');
      console.log('  1. Completed the full login process');
      console.log('  2. Can see the home timeline');
      console.log('\nYou may need to handle 2FA or verification challenges.');

      // Let user try again
      console.log('\nThe browser will stay open. Try logging in again.');
      await askQuestion('Press ENTER when ready to check again... ');

      await page.goto('https://x.com/home', { waitUntil: 'networkidle2', timeout: 30000 });
      await sleep(2000);

      const retryLoggedIn = await isLoggedIn(page);
      if (retryLoggedIn) {
        console.log('\n✓ Login verified on retry!');
        await saveSession(page);
        console.log('✓ Session saved!');
      } else {
        console.log('\n✗ Still cannot verify login.');
        console.log('Please try running this script again and complete the full login.\n');
      }
    }

  } catch (error) {
    console.error('\nError:', error.message);
  } finally {
    if (browser) {
      console.log('\nClosing browser in 5 seconds...');
      await sleep(5000);
      await browser.close();
    }
  }
}

main();
