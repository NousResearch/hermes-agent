
require('dotenv').config({ path: '../../.env' });
const { ethers } = require('ethers');

async function generatePairingUrl() {
    console.log("Menghubungkan ke Billions Network...");
    
    try {
        // 1. Ambil Private Key dari .env
        const privateKey = process.env.WALLET_PRIVATE_KEY;
        if (!privateKey || privateKey === "0xffc9954d70c446be37676fade5eeea853846eef93c496472f560d0174bf4a8c2") {
            throw new Error("Private Key belum diisi dengan benar di file .env bro!");
        }

        // 2. Setup Wallet pakai Ethers.js
        const wallet = new ethers.Wallet(privateKey);
        console.log(`[+] Wallet Agent Terdeteksi: ${wallet.address}`);

        // 3. Bikin Signature buat Bukti Kepemilikan KYA
        // Timestamp ditambahkan biar signature selalu unik (mencegah replay attack)
        const timestamp = Date.now();
        const message = `Authenticate_Hermes_Agent_to_Billions_${timestamp}`;
        const signature = await wallet.signMessage(message);

        // 4. Generate URL Pairing Billions Network
        // Parameter framework diset 'hermes' untuk ngebuktiin ke sistem mereka
        const pairingUrl = `https://app.billions.network/pair?agentAddress=${wallet.address}&signature=${signature}&timestamp=${timestamp}&framework=hermes`;

        console.log("\n==================================================");
        console.log("🚀 STATUS: IDENTITAS AGENT BERHASIL DIBUAT 🚀");
        console.log("==================================================");
        console.log("Link Paired Agent Lu (Copas ke Browser):");
        console.log(`➡️  ${pairingUrl}`);
        console.log("==================================================\n");
        
        console.log("Menunggu verifikasi on-chain...");

    } catch (error) {
        console.error("[-] Waduh, ada error nih bro:", error.message);
    }
}

generatePairingUrl();
