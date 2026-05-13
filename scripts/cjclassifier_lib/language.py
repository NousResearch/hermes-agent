# Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language enum for CJClassifier."""

from enum import IntEnum
from typing import Dict, List, Tuple


class CJLanguage(IntEnum):
    """Represents the three CJ languages plus UNKNOWN.

    The int value is the internal array index used by the classifier:
    CHINESE_SIMPLIFIED=0, CHINESE_TRADITIONAL=1, JAPANESE=2, UNKNOWN=-1.
    """

    UNKNOWN = -1
    CHINESE_SIMPLIFIED = 0
    CHINESE_TRADITIONAL = 1
    JAPANESE = 2

    @property
    def iso_code(self) -> str:
        """ISO 639-1-style code (e.g. 'zh-hans', 'ja')."""
        return _LANG_INFO[self][0]

    @property
    def iso_code3(self) -> str:
        """ISO 639-3-style code (e.g. 'zho-hans', 'jpn')."""
        return _LANG_INFO[self][1]

    @property
    def alt_names(self) -> List[str]:
        """Alternate name strings recognized by from_string()."""
        return list(_LANG_INFO[self][2])

    def is_chinese(self) -> bool:
        """True if this is CHINESE_SIMPLIFIED or CHINESE_TRADITIONAL."""
        return self in (CJLanguage.CHINESE_SIMPLIFIED, CJLanguage.CHINESE_TRADITIONAL)

    def is_japanese(self) -> bool:
        """True if this is JAPANESE."""
        return self is CJLanguage.JAPANESE

    @classmethod
    def from_string(cls, s: str) -> "CJLanguage":
        """Look up a CJLanguage by name, ISO code, or alternate name (case-insensitive).

        Returns UNKNOWN if not recognized.
        """
        if s is None:
            return cls.UNKNOWN
        return _BY_NAME.get(s.lower(), cls.UNKNOWN)


# Per-language metadata: (iso_code, iso_code3, alt_names)
_LANG_INFO: Dict[CJLanguage, Tuple[str, str, List[str]]] = {
    CJLanguage.UNKNOWN: ("", "", []),
    CJLanguage.CHINESE_SIMPLIFIED: (
        "zh-hans", "zho-hans",
        ["chinese", "zh", "zh-cn", "zh-hans-cn", "zh-hans-sg"],
    ),
    CJLanguage.CHINESE_TRADITIONAL: (
        "zh-hant", "zho-hant",
        ["zh-hant-hk", "zh-hk", "zh-hant-tw"],
    ),
    CJLanguage.JAPANESE: ("ja", "jpn", ["jp"]),
}

# Build the lookup table once at module load time.
_BY_NAME: Dict[str, CJLanguage] = {}
for _lang in CJLanguage:
    _BY_NAME[_lang.name.lower()] = _lang
    if _lang.iso_code:
        _BY_NAME[_lang.iso_code] = _lang
    if _lang.iso_code3:
        _BY_NAME[_lang.iso_code3] = _lang
    for _name in _lang.alt_names:
        _BY_NAME[_name] = _lang
