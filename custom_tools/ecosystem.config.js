/**
 * PM2 Ecosystem Configuration - Hermes Web3 Tools
 * =================================================
 * 
 * Start all services:
 *   pm2 start custom_tools/ecosystem.config.js
 * 
 * Start individual:
 *   pm2 start custom_tools/ecosystem.config.js --only hermes-telegram-bot
 * 
 * Manage:
 *   pm2 status
 *   pm2 logs hermes-telegram-bot
 *   pm2 restart hermes-telegram-bot
 *   pm2 stop all
 * 
 * Persist across reboots:
 *   pm2 save
 *   pm2 startup
 */

module.exports = {
  apps: [
    {
      name: "hermes-telegram-bot",
      script: "python",
      args: "-m custom_tools.telegram_gateway.bot",
      cwd: process.env.HERMES_DIR || ".",
      interpreter: "none",
      env: {
        TELEGRAM_BOT_TOKEN: process.env.TELEGRAM_BOT_TOKEN || "",
        TELEGRAM_ALLOWED_USERS: process.env.TELEGRAM_ALLOWED_USERS || "",
        DRY_RUN: "true",
        ETH_RPC_URL: process.env.ETH_RPC_URL || "",
        BASE_RPC_URL: process.env.BASE_RPC_URL || "",
        ARB_RPC_URL: process.env.ARB_RPC_URL || "",
        POLYGON_RPC_URL: process.env.POLYGON_RPC_URL || "",
        ETHERSCAN_API_KEY: process.env.ETHERSCAN_API_KEY || "",
        APPROVAL_DB_DIR: ".data",
        WALLETS_DIR: ".wallets",
      },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "logs/telegram-bot-error.log",
      out_file: "logs/telegram-bot-out.log",
      merge_logs: true,
    },
  ],
};
