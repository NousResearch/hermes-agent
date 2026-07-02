/* Bit and bytes utilities.

   Bytes swap functions, reverse order of bytes:

   - _Py_bswap16(uint16_t)
   - _Py_bswap32(uint32_t)
   - _Py_bswap64(uint64_t)
*/

#ifndef Py_INTERNAL_BITUTILS_H
#define Py_INTERNAL_BITUTILS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#if defined(__GNUC__) \
      && ((__GNUC__ >= 5) || (__GNUC__ == 4) && (__GNUC_MINOR__ >= 8))
   /* __builtin_bswap16() is available since GCC 4.8,
      __builtin_bswap32() is available since GCC 4.3,
      __builtin_bswap64() is available since GCC 4.3. */
#  define _PY_HAVE_BUILTIN_BSWAP
#endif

#ifdef _MSC_VER
   /* Get _byteswap_ushort(), _byteswap_ulong(), _byteswap_uint64() */
#  include <intrin.h>
#endif

static inline uint16_t
_Py_bswap16(uint16_t word)
{
#if defined(_PY_HAVE_BUILTIN_BSWAP) || _Py__has_builtin(__builtin_bswap16)
    return __builtin_bswap16(word);
#elif defined(_MSC_VER)
    Py_BUILD_ASSERT(sizeof(word) == sizeof(unsigned short));
    return _byteswap_ushort(word);
#else
    // Portable implementation which doesn't rely on circular bit shift
    return ( ((word & UINT16_C(0x00FF)) << 8)
           | ((word & UINT16_C(0xFF00)) >> 8));
#endif
}

static inline uint32_t
_Py_bswap32(uint32_t word)
{
#if defined(_PY_HAVE_BUILTIN_BSWAP) || _Py__has_builtin(__builtin_bswap32)
    return __builtin_bswap32(word);
#elif defined(_MSC_VER)
    Py_BUILD_ASSERT(sizeof(word) == sizeof(unsigned long));
    return _byteswap_ulong(word);
#else
    // Portable implementation which doesn't rely on circular bit shift
    return ( ((word & UINT32_C(0x000000FF)) << 24)
           | ((word & UINT32_C(0x0000FF00)) <<  8)
           | ((word & UINT32_C(0x00FF0000)) >>  8)
           | ((word & UINT32_C(0xFF000000)) >> 24));
#endif
}

static inline uint64_t
_Py_bswap64(uint64_t word)
{
#if defined(_PY_HAVE_BUILTIN_BSWAP) || _Py__has_builtin(__builtin_bswap64)
    return __builtin_bswap64(word);
#elif defined(_MSC_VER)
    return _byteswap_uint64(word);
#else
    // Portable implementation which doesn't rely on circular bit shift
    return ( ((word & UINT64_C(0x00000000000000FF)) << 56)
           | ((word & UINT64_C(0x000000000000FF00)) << 40)
           | ((word & UINT64_C(0x0000000000FF0000)) << 24)
           | ((word & UINT64_C(0x00000000FF000000)) <<  8)
           | ((word & UINT64_C(0x000000FF00000000)) >>  8)
           | ((word & UINT64_C(0x0000FF0000000000)) >> 24)
           | ((word & UINT64_C(0x00FF000000000000)) >> 40)
           | ((word & UINT64_C(0xFF00000000000000)) >> 56));
#endif
}


// Population count: count the number of 1's in 'x'
// (number of bits set to 1), also known as the hamming weight.
//
// Implementation note. CPUID is not used, to test if x86 POPCNT instruction
// can be used, to keep the implementation simple. For example, Visual Studio
// __popcnt() is not used this reason. The clang and GCC builtin function can
// use the x86 POPCNT instruction if the target architecture has SSE4a or
// newer.
static inline int
_Py_popcount32(uint32_t x)
{
#if (defined(__clang__) || defined(__GNUC__))

#if SIZEOF_INT >= 4
    Py_BUILD_ASSERT(sizeof(x) <= sizeof(unsigned int));
    return __builtin_popcount(x);
#else
    // The C standard guarantees that unsigned long will always be big enough
    // to hold a uint32_t value without losing information.
    Py_BUILD_ASSERT(sizeof(x) <= sizeof(unsigned long));
    return __builtin_popcountl(x);
#endif

#else
    // 32-bit SWAR (SIMD Within A Register) popcount

    // Binary: 0 1 0 1 ...
    const uint32_t M1 = 0x55555555;
    // Binary: 00 11 00 11. ..
    const uint32_t M2 = 0x33333333;
    // Binary: 0000 1111 0000 1111 ...
    const uint32_t M4 = 0x0F0F0F0F;

    // Put count of each 2 bits into those 2 bits
    x = x - ((x >> 1) & M1);
    // Put count of each 4 bits into those 4 bits
    x = (x & M2) + ((x >> 2) & M2);
    // Put count of each 8 bits into those 8 bits
    x = (x + (x >> 4)) & M4;
    // Sum of the 4 byte counts.
    // Take care when considering changes to the next line. Portability and
    // correctness are delicate here, thanks to C's "integer promotions" (C99
    // §6.3.1.1p2). On machines where the `int` type has width greater than 32
    // bits, `x` will be promoted to an `int`, and following C's "usual
    // arithmetic conversions" (C99 §6.3.1.8), the multiplication will be
    // performed as a multiplication of two `unsigned int` operands. In this
    // case it's critical that we cast back to `uint32_t` in order to keep only
    // the least significant 32 bits. On machines where the `int` type has
    // width no greater than 32, the multiplication is of two 32-bit unsigned
    // integer types, and the (uint32_t) cast is a no-op. In both cases, we
    // avoid the risk of undefined behaviour due to overflow of a
    // multiplication of signed integer types.
    return (uint32_t)(x * 0x01010101U) >> 24;
#endif
}


// Return the index of the most significant 1 bit in 'x'. This is the smallest
// integer k such that x < 2**k. Equivalent to floor(log2(x)) + 1 for x != 0.
static inline int
_Py_bit_length(unsigned long x)
{
#if (defined(__clang__) || defined(__GNUC__))
    if (x != 0) {
        // __builtin_clzl() is available since GCC 3.4.
        // Undefined behavior for x == 0.
        return (int)sizeof(unsigned long) * 8 - __builtin_clzl(x);
    }
    else {
        return 0;
    }
#elif defined(_MSC_VER)
    // _BitScanReverse() is documented to search 32 bits.
    Py_BUILD_ASSERT(sizeof(unsigned long) <= 4);
    unsigned long msb;
    if (_BitScanReverse(&msb, x)) {
        return (int)msb + 1;
    }
    else {
        return 0;
    }
#else
    const int BIT_LENGTH_TABLE[32] = {
        0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
    };
    int msb = 0;
    while (x >= 32) {
        msb += 6;
        x >>= 6;
    }
    msb += BIT_LENGTH_TABLE[x];
    return msb;
#endif
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_BITUTILS_H */
