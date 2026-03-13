require('dotenv').config();
const { ethers } = require('ethers');

async function generateLongLink() {
    try {
        console.log("Membaca identitas dari .env...");
        const privateKey = process.env.WALLET_PRIVATE_KEY;
        
        if (!privateKey) {
            throw new Error("Private Key nggak ketemu di .env!");
        }

        const wallet = new ethers.Wallet(privateKey);
        console.log(`[+] Wallet Agent: ${wallet.address}`);
        
        // Bikin tantangan/pesan unik
        const timestamp = Date.now();
        const message = `KYA_Verification_Hermes_Agent_${timestamp}`;
        const signature = await wallet.signMessage(message);

        // Susun data mentah sesuai format Web3 KYA
        const payloadData = {
            agentAddress: wallet.address,
            signature: signature,
            timestamp: timestamp,
            framework: "hermes",
            metadata: {
                name: "Hermes Agent",
                developer: "AgungPrabowo123",
                description: "Hermes Agent Identity Bridge Custom Integration"
            }
        };

        // Enkripsi/Encode data jadi Base64 biar panjang dan aman
        const base64Payload = Buffer.from(JSON.stringify(payloadData)).toString('base64');
        
        // Rakit URL akhirnya
        const pairingUrl = `https://app.billions.network/pair?payload=${base64Payload}`;

        console.log("\n=====================================================================");
        console.log("🚀 BOOM! LINK PAIRED AGENT (FORMAT PANJANG BASE64) BERHASIL DIBUAT 🚀");
        console.log("=====================================================================");
        console.log(pairingUrl);
        console.log("=====================================================================\n");
        console.log("Silakan copas link di atas ke browser lu bro!");

    } catch (err) {
        console.error("[-] Waduh, error bro:", err.message);
    }
}

generateLongLink();
