// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * DropClaw Marketplace V1 — Decentralized File Marketplace
 *
 * Agents list encrypted files for sale. Buyers pay MON.
 * Commit-reveal key exchange with escrow.
 * Fully immutable — no admin, no pause, no upgrade.
 */
contract MarketplaceV1 {
    // ═══════════ CONSTANTS ═══════════
    uint256 public constant FEE_BPS = 250; // 2.5%
    uint256 public constant KEY_DELIVERY_TIMEOUT = 24 hours;
    address payable public immutable treasury;

    // ═══════════ STRUCTS ═══════════
    struct Listing {
        address payable seller;
        string fileId;
        string title;
        string description;
        string skillFileUri;  // IPFS CID for skill file JSON
        bytes32 keyHash;      // keccak256(abi.encodePacked(keyHex))
        uint256 price;        // in wei (MON)
        uint256 createdAt;
        address buyer;
        uint256 purchasedAt;
        bool keyRevealed;
        bool refunded;
        bool active;
    }

    // ═══════════ STATE ═══════════
    uint256 public listingCount;
    mapping(uint256 => Listing) public listings;

    // ═══════════ EVENTS ═══════════
    event Listed(
        uint256 indexed listingId,
        address indexed seller,
        string fileId,
        string title,
        uint256 price
    );
    event Purchased(
        uint256 indexed listingId,
        address indexed buyer,
        uint256 price
    );
    event KeyRevealed(
        uint256 indexed listingId,
        string key
    );
    event Refunded(
        uint256 indexed listingId,
        address indexed buyer,
        uint256 amount
    );
    event Delisted(
        uint256 indexed listingId
    );

    // ═══════════ CONSTRUCTOR ═══════════
    constructor(address payable _treasury) {
        require(_treasury != address(0), "Invalid treasury");
        treasury = _treasury;
    }

    // ═══════════ MODIFIERS ═══════════
    modifier listingExists(uint256 _listingId) {
        require(_listingId < listingCount, "Listing does not exist");
        _;
    }

    // ═══════════ FUNCTIONS ═══════════

    /**
     * @notice List a file for sale
     * @param _fileId DropClaw file ID
     * @param _title Display title
     * @param _description File description
     * @param _skillFileUri IPFS CID of the skill file JSON
     * @param _keyHash keccak256(abi.encodePacked(keyHex)) — commit phase
     * @param _price Price in wei (MON)
     */
    function listFile(
        string calldata _fileId,
        string calldata _title,
        string calldata _description,
        string calldata _skillFileUri,
        bytes32 _keyHash,
        uint256 _price
    ) external returns (uint256) {
        require(bytes(_fileId).length > 0, "Empty fileId");
        require(bytes(_title).length > 0, "Empty title");
        require(_keyHash != bytes32(0), "Invalid keyHash");
        require(_price > 0, "Price must be > 0");

        uint256 listingId = listingCount;
        listings[listingId] = Listing({
            seller: payable(msg.sender),
            fileId: _fileId,
            title: _title,
            description: _description,
            skillFileUri: _skillFileUri,
            keyHash: _keyHash,
            price: _price,
            createdAt: block.timestamp,
            buyer: address(0),
            purchasedAt: 0,
            keyRevealed: false,
            refunded: false,
            active: true
        });

        listingCount++;

        emit Listed(listingId, msg.sender, _fileId, _title, _price);
        return listingId;
    }

    /**
     * @notice Purchase a listed file — MON escrowed in contract
     */
    function purchase(uint256 _listingId) external payable listingExists(_listingId) {
        Listing storage l = listings[_listingId];
        require(l.active, "Listing not active");
        require(l.buyer == address(0), "Already purchased");
        require(msg.sender != l.seller, "Seller cannot buy own listing");
        require(msg.value == l.price, "Incorrect payment amount");

        l.buyer = msg.sender;
        l.purchasedAt = block.timestamp;

        emit Purchased(_listingId, msg.sender, msg.value);
    }

    /**
     * @notice Seller delivers the decryption key — reveal phase
     * @param _listingId Listing ID
     * @param _keyHex The decryption key (hex string)
     */
    function deliverKey(uint256 _listingId, string calldata _keyHex) external listingExists(_listingId) {
        Listing storage l = listings[_listingId];
        require(msg.sender == l.seller, "Only seller");
        require(l.buyer != address(0), "Not purchased");
        require(!l.keyRevealed, "Key already revealed");
        require(!l.refunded, "Already refunded");

        // Verify commit: keccak256(keyHex) must match keyHash
        require(
            keccak256(abi.encodePacked(_keyHex)) == l.keyHash,
            "Key does not match committed hash"
        );

        // Effects
        l.keyRevealed = true;

        // Calculate fee
        uint256 fee = (l.price * FEE_BPS) / 10000;
        uint256 sellerPayout = l.price - fee;

        // Interactions — checks-effects-interactions pattern
        (bool sentSeller, ) = l.seller.call{value: sellerPayout}("");
        require(sentSeller, "Seller payment failed");

        if (fee > 0) {
            (bool sentTreasury, ) = treasury.call{value: fee}("");
            require(sentTreasury, "Treasury payment failed");
        }

        emit KeyRevealed(_listingId, _keyHex);
    }

    /**
     * @notice Buyer claims refund if seller doesn't deliver key within timeout
     */
    function claimRefund(uint256 _listingId) external listingExists(_listingId) {
        Listing storage l = listings[_listingId];
        require(msg.sender == l.buyer, "Only buyer");
        require(l.buyer != address(0), "Not purchased");
        require(!l.keyRevealed, "Key was delivered");
        require(!l.refunded, "Already refunded");
        require(
            block.timestamp >= l.purchasedAt + KEY_DELIVERY_TIMEOUT,
            "Timeout not reached"
        );

        // Effects
        l.refunded = true;

        // Interactions
        (bool sent, ) = payable(l.buyer).call{value: l.price}("");
        require(sent, "Refund failed");

        emit Refunded(_listingId, l.buyer, l.price);
    }

    /**
     * @notice Seller removes an unsold listing
     */
    function delistFile(uint256 _listingId) external listingExists(_listingId) {
        Listing storage l = listings[_listingId];
        require(msg.sender == l.seller, "Only seller");
        require(l.buyer == address(0), "Cannot delist after purchase");
        require(l.active, "Already delisted");

        l.active = false;

        emit Delisted(_listingId);
    }

    /**
     * @notice Get full listing details (view)
     */
    function getListing(uint256 _listingId)
        external
        view
        listingExists(_listingId)
        returns (Listing memory)
    {
        return listings[_listingId];
    }
}
