import sqlite3

import pytest

from plugins.platforms.discord.adapter import (
    _league_mmr_fetch_leaderboard,
    _league_mmr_format_leaderboard,
    _league_mmr_init_db,
    _league_mmr_parse_riot_id,
    _league_mmr_upsert_player,
)


def test_league_mmr_parse_riot_id_accepts_spaces():
    assert _league_mmr_parse_riot_id("Pope Francis#bless") == ("Pope Francis", "bless")


@pytest.mark.parametrize("value", ["Pope Francis", "#EUW", "Edison#", "A#EUW", "Edison#EUROPELONG"])
def test_league_mmr_parse_riot_id_rejects_invalid(value):
    with pytest.raises(ValueError):
        _league_mmr_parse_riot_id(value)


def test_league_mmr_upsert_player_preserves_rating_stats():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _league_mmr_init_db(conn)
    _league_mmr_upsert_player(
        conn,
        discord_user_id="123",
        discord_display_name="Edison",
        riot_game_name="Pope Francis",
        riot_tag_line="bless",
    )
    conn.execute(
        "UPDATE league_mmr_players SET mmr = 1042, games_played = 3, wins = 2, losses = 1 WHERE discord_user_id = '123'"
    )
    _league_mmr_upsert_player(
        conn,
        discord_user_id="123",
        discord_display_name="Edison2",
        riot_game_name="New Name",
        riot_tag_line="EUW",
    )

    row = conn.execute("SELECT * FROM league_mmr_players WHERE discord_user_id = '123'").fetchone()
    assert row["riot_game_name"] == "New Name"
    assert row["riot_tag_line"] == "EUW"
    assert row["mmr"] == 1042
    assert row["games_played"] == 3
    assert row["wins"] == 2
    assert row["losses"] == 1


def test_league_mmr_upsert_player_merges_manual_row_when_user_registers_later():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _league_mmr_init_db(conn)

    _league_mmr_upsert_player(
        conn,
        guild_id="guild-1",
        discord_user_id="manual:pope-francis-bless",
        discord_display_name="Pope Francis#BLESS",
        riot_game_name="Pope Francis",
        riot_tag_line="BLESS",
    )
    conn.execute(
        """
        UPDATE league_mmr_players
        SET mmr = 1021, games_played = 3, wins = 2, losses = 1
        WHERE discord_user_id = 'manual:pope-francis-bless'
        """
    )

    _league_mmr_upsert_player(
        conn,
        guild_id="guild-1",
        discord_user_id="987654321",
        discord_display_name="Real Discord User",
        riot_game_name="Pope Francis",
        riot_tag_line="bless",
    )

    rows = conn.execute("SELECT * FROM league_mmr_players").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["discord_user_id"] == "987654321"
    assert row["discord_display_name"] == "Real Discord User"
    assert row["mmr"] == 1021
    assert row["games_played"] == 3
    assert row["wins"] == 2
    assert row["losses"] == 1


def test_league_mmr_fetch_leaderboard_orders_by_mmr_and_filters_guild():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _league_mmr_init_db(conn)
    players = [
        ("guild-1", "1", "A", "Alpha", "EUW", 1010, 4, 2, 2),
        ("guild-1", "2", "B", "Bravo", "EUW", 1050, 2, 2, 0),
        ("guild-2", "3", "C", "Charlie", "EUW", 2000, 1, 1, 0),
        ("", "4", "D", "Delta", "EUW", 990, 0, 0, 0),
    ]
    conn.executemany(
        """
        INSERT INTO league_mmr_players (
            guild_id, discord_user_id, discord_display_name,
            riot_game_name, riot_tag_line, mmr, games_played, wins, losses
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        players,
    )

    rows = _league_mmr_fetch_leaderboard(conn, guild_id="guild-1", limit=10)

    assert [row["riot_game_name"] for row in rows] == ["Bravo", "Alpha", "Delta"]


def test_league_mmr_fetch_leaderboard_default_returns_all_players():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _league_mmr_init_db(conn)
    conn.executemany(
        """
        INSERT INTO league_mmr_players (
            guild_id, discord_user_id, discord_display_name,
            riot_game_name, riot_tag_line, mmr, games_played, wins, losses
        ) VALUES ('guild-1', ?, ?, ?, 'EUW', 1000, 0, 0, 0)
        """,
        [(str(i), f"Player {i}", f"Player{i}") for i in range(12)],
    )

    rows = _league_mmr_fetch_leaderboard(conn, guild_id="guild-1")

    assert len(rows) == 12


def test_league_mmr_format_leaderboard_includes_rank_mmr_and_winrate():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _league_mmr_init_db(conn)
    conn.execute(
        """
        INSERT INTO league_mmr_players (
            guild_id, discord_user_id, discord_display_name,
            riot_game_name, riot_tag_line, mmr, games_played, wins, losses
        ) VALUES ('guild-1', '1', 'Edison', 'Pope Francis', 'BLESS', 1042, 4, 3, 1)
        """
    )
    rows = _league_mmr_fetch_leaderboard(conn, guild_id="guild-1", limit=10)

    message = _league_mmr_format_leaderboard(rows)

    assert "🏆 League Customs MMR Leaderboard" in message
    assert "**1.** `Pope Francis#BLESS` — **1042 MMR**" in message
    assert "3W/1L, 4 games, 75% WR" in message


def test_league_mmr_format_leaderboard_empty_state():
    assert "Ingen spillere er registrert" in _league_mmr_format_leaderboard([])
