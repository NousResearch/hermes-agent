/*
   _BlocksOutputBuffer is used to maintain an output buffer
   that has unpredictable size. Suitable for compression/decompression
   API (bz2/lzma/zlib) that has stream->next_out and stream->avail_out:

        stream->next_out:  point to the next output position.
        stream->avail_out: the number of available bytes left in the buffer.

   It maintains a list of bytes object, so there is no overhead of resizing
   the buffer.

   Usage:

   1, Initialize the struct instance like this:
        _BlocksOutputBuffer buffer = {.list = NULL};
      Set .list to NULL for _BlocksOutputBuffer_OnError()

   2, Initialize the buffer use one of these functions:
        _BlocksOutputBuffer_InitAndGrow()
        _BlocksOutputBuffer_InitWithSize()

   3, If (avail_out == 0), grow the buffer:
        _BlocksOutputBuffer_Grow()

   4, Get the current outputted data size:
        _BlocksOutputBuffer_GetDataSize()

   5, Finish the buffer, and return a bytes object:
        _BlocksOutputBuffer_Finish()

   6, Clean up the buffer when an error occurred:
        _BlocksOutputBuffer_OnError()
*/

#ifndef Py_INTERNAL_BLOCKS_OUTPUT_BUFFER_H
#define Py_INTERNAL_BLOCKS_OUTPUT_BUFFER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"

typedef struct {
    // List of bytes objects
    PyObject *list;
    // Number of whole allocated size
    Py_ssize_t allocated;
    // Max length of the buffer, negative number means unlimited length.
    Py_ssize_t max_length;
} _BlocksOutputBuffer;

static const char unable_allocate_msg[] = "Unable to allocate output buffer.";

/* In 32-bit build, the max block size should <= INT32_MAX. */
#define OUTPUT_BUFFER_MAX_BLOCK_SIZE (256*1024*1024)

/* Block size sequence */
#define KB (1024)
#define MB (1024*1024)
static const Py_ssize_t BUFFER_BLOCK_SIZE[] =
    { 32*KB, 64*KB, 256*KB, 1*MB, 4*MB, 8*MB, 16*MB, 16*MB,
      32*MB, 32*MB, 32*MB, 32*MB, 64*MB, 64*MB, 128*MB, 128*MB,
      OUTPUT_BUFFER_MAX_BLOCK_SIZE };
#undef KB
#undef MB

/* According to the block sizes defined by BUFFER_BLOCK_SIZE, the whole
   allocated size growth step is:
    1   32 KB       +32 KB
    2   96 KB       +64 KB
    3   352 KB      +256 KB
    4   1.34 MB     +1 MB
    5   5.34 MB     +4 MB
    6   13.34 MB    +8 MB
    7   29.34 MB    +16 MB
    8   45.34 MB    +16 MB
    9   77.34 MB    +32 MB
    10  109.34 MB   +32 MB
    11  141.34 MB   +32 MB
    12  173.34 MB   +32 MB
    13  237.34 MB   +64 MB
    14  301.34 MB   +64 MB
    15  429.34 MB   +128 MB
    16  557.34 MB   +128 MB
    17  813.34 MB   +256 MB
    18  1069.34 MB  +256 MB
    19  1325.34 MB  +256 MB
    20  1581.34 MB  +256 MB
    21  1837.34 MB  +256 MB
    22  2093.34 MB  +256 MB
    ...
*/

/* Initialize the buffer, and grow the buffer.

   max_length: Max length of the buffer, -1 for unlimited length.

   On success, return allocated size (>=0)
   On failure, return -1
*/
static inline Py_ssize_t
_BlocksOutputBuffer_InitAndGrow(_BlocksOutputBuffer *buffer,
                                const Py_ssize_t max_length,
                                void **next_out)
{
    PyObject *b;
    Py_ssize_t block_size;

    // ensure .list was set to NULL
    assert(buffer->list == NULL);

    // get block size
    if (0 <= max_length && max_length < BUFFER_BLOCK_SIZE[0]) {
        block_size = max_length;
    } else {
        block_size = BUFFER_BLOCK_SIZE[0];
    }

    // the first block
    b = PyBytes_FromStringAndSize(NULL, block_size);
    if (b == NULL) {
        return -1;
    }

    // create the list
    buffer->list = PyList_New(1);
    if (buffer->list == NULL) {
        Py_DECREF(b);
        return -1;
    }
    PyList_SET_ITEM(buffer->list, 0, b);

    // set variables
    buffer->allocated = block_size;
    buffer->max_length = max_length;

    *next_out = PyBytes_AS_STRING(b);
    return block_size;
}

/* Initialize the buffer, with an initial size.

   Check block size limit in the outer wrapper function. For example, some libs
   accept UINT32_MAX as the maximum block size, then init_size should <= it.

   On success, return allocated size (>=0)
   On failure, return -1
*/
static inline Py_ssize_t
_BlocksOutputBuffer_InitWithSize(_BlocksOutputBuffer *buffer,
                                 const Py_ssize_t init_size,
                                 void **next_out)
{
    PyObject *b;

    // ensure .list was set to NULL
    assert(buffer->list == NULL);

    // the first block
    b = PyBytes_FromStringAndSize(NULL, init_size);
    if (b == NULL) {
        PyErr_SetString(PyExc_MemoryError, unable_allocate_msg);
        return -1;
    }

    // create the list
    buffer->list = PyList_New(1);
    if (buffer->list == NULL) {
        Py_DECREF(b);
        return -1;
    }
    PyList_SET_ITEM(buffer->list, 0, b);

    // set variables
    buffer->allocated = init_size;
    buffer->max_length = -1;

    *next_out = PyBytes_AS_STRING(b);
    return init_size;
}

/* Grow the buffer. The avail_out must be 0, please check it before calling.

   On success, return allocated size (>=0)
   On failure, return -1
*/
static inline Py_ssize_t
_BlocksOutputBuffer_Grow(_BlocksOutputBuffer *buffer,
                         void **next_out,
                         const Py_ssize_t avail_out)
{
    PyObject *b;
    const Py_ssize_t list_len = Py_SIZE(buffer->list);
    Py_ssize_t block_size;

    // ensure no gaps in the data
    if (avail_out != 0) {
        PyErr_SetString(PyExc_SystemError,
                        "avail_out is non-zero in _BlocksOutputBuffer_Grow().");
        return -1;
    }

    // get block size
    if (list_len < (Py_ssize_t) Py_ARRAY_LENGTH(BUFFER_BLOCK_SIZE)) {
        block_size = BUFFER_BLOCK_SIZE[list_len];
    } else {
        block_size = BUFFER_BLOCK_SIZE[Py_ARRAY_LENGTH(BUFFER_BLOCK_SIZE) - 1];
    }

    // check max_length
    if (buffer->max_length >= 0) {
        // if (rest == 0), should not grow the buffer.
        Py_ssize_t rest = buffer->max_length - buffer->allocated;
        assert(rest > 0);

        // block_size of the last block
        if (block_size > rest) {
            block_size = rest;
        }
    }

    // check buffer->allocated overflow
    if (block_size > PY_SSIZE_T_MAX - buffer->allocated) {
        PyErr_SetString(PyExc_MemoryError, unable_allocate_msg);
        return -1;
    }

    // create the block
    b = PyBytes_FromStringAndSize(NULL, block_size);
    if (b == NULL) {
        PyErr_SetString(PyExc_MemoryError, unable_allocate_msg);
        return -1;
    }
    if (PyList_Append(buffer->list, b) < 0) {
        Py_DECREF(b);
        return -1;
    }
    Py_DECREF(b);

    // set variables
    buffer->allocated += block_size;

    *next_out = PyBytes_AS_STRING(b);
    return block_size;
}

/* Return the current outputted data size. */
static inline Py_ssize_t
_BlocksOutputBuffer_GetDataSize(_BlocksOutputBuffer *buffer,
                                const Py_ssize_t avail_out)
{
    return buffer->allocated - avail_out;
}

/* Finish the buffer.

   Return a bytes object on success
   Return NULL on failure
*/
static inline PyObject *
_BlocksOutputBuffer_Finish(_BlocksOutputBuffer *buffer,
                           const Py_ssize_t avail_out)
{
    PyObject *result, *block;
    const Py_ssize_t list_len = Py_SIZE(buffer->list);

    // fast path for single block
    if ((list_len == 1 && avail_out == 0) ||
        (list_len == 2 && Py_SIZE(PyList_GET_ITEM(buffer->list, 1)) == avail_out))
    {
        block = PyList_GET_ITEM(buffer->list, 0);
        Py_INCREF(block);

        Py_CLEAR(buffer->list);
        return block;
    }

    // final bytes object
    result = PyBytes_FromStringAndSize(NULL, buffer->allocated - avail_out);
    if (result == NULL) {
        PyErr_SetString(PyExc_MemoryError, unable_allocate_msg);
        return NULL;
    }

    // memory copy
    if (list_len > 0) {
        char *posi = PyBytes_AS_STRING(result);

        // blocks except the last one
        Py_ssize_t i = 0;
        for (; i < list_len-1; i++) {
            block = PyList_GET_ITEM(buffer->list, i);
            memcpy(posi, PyBytes_AS_STRING(block), Py_SIZE(block));
            posi += Py_SIZE(block);
        }
        // the last block
        block = PyList_GET_ITEM(buffer->list, i);
        memcpy(posi, PyBytes_AS_STRING(block), Py_SIZE(block) - avail_out);
    } else {
        assert(Py_SIZE(result) == 0);
    }

    Py_CLEAR(buffer->list);
    return result;
}

/* Clean up the buffer when an error occurred. */
static inline void
_BlocksOutputBuffer_OnError(_BlocksOutputBuffer *buffer)
{
    Py_CLEAR(buffer->list);
}

#ifdef __cplusplus
}
#endif
#endif /* Py_INTERNAL_BLOCKS_OUTPUT_BUFFER_H */